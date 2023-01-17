import numpy as np

import pandas as pd 

from pathlib import Path

import json



from fastprogress import progress_bar

from tqdm.auto import tqdm



tqdm.pandas()
ROOT_PATH = Path("../input")

TRAIN_IMAGE_PATH = ROOT_PATH/"train_images"

TEST_IMAGE_PATH = ROOT_PATH/"test_images"

WORKING_PATH = Path("../working")
train_data = pd.read_csv(ROOT_PATH/"train/train.csv")

test_data = pd.read_csv(ROOT_PATH/"test/test.csv")



state_labels = pd.read_csv(ROOT_PATH/"state_labels.csv")

color_labels = pd.read_csv(ROOT_PATH/"color_labels.csv")

breed_labels = pd.read_csv(ROOT_PATH/"breed_labels.csv")
print(f"Train set length: {len(train_data)}, test set length: {len(test_data)}")
train_data.head(4).transpose()
train_data.describe().transpose()
import codecs



def add_sentiment_and_magnitude(dataset, json_path):

    magnitude = []

    sentiment = []

    

    for pet_id in tqdm(dataset['PetID']):

        try:

            with codecs.open(json_path/(pet_id + ".json"), encoding='UTF-8') as json_file:

                doc_sentiment = json.load(json_file)['documentSentiment']

                magnitude.append(doc_sentiment['magnitude'])

                sentiment.append(doc_sentiment['score'])

        except FileNotFoundError:

            magnitude.append(np.nan)

            sentiment.append(np.nan)

            

    return pd.DataFrame({'PetID':dataset['PetID'], 'magnitude': magnitude, 'sentiment':sentiment})



def expand_with_sentiment(dataset, json_path=ROOT_PATH/"train_sentiment/"):

    sentiment_data = add_sentiment_and_magnitude(dataset, json_path)

    return pd.merge(dataset, sentiment_data, on='PetID')



def add_no_photo_feature():

    pass
train_data = expand_with_sentiment(train_data)

test_data = expand_with_sentiment(test_data, json_path=ROOT_PATH/"test_sentiment/")    
# Creating a validation set



valid_data = train_data[-2000:]

# train_data = train_data[:-2000]

train_data = train_data
from fastai.tabular import *



procedures = [FillMissing, Categorify, Normalize]

valid_id = range(len(train_data) - 2000, len(train_data))



dependant_var = 'AdoptionSpeed'

not_used_cols = ['RescuerID', 'Description', 'PetID', 'Breed2', 'Color3']

used_cols = [col for col in train_data.columns if col not in not_used_cols]

continious_cols = ['Age', 'Fee', 'magnitude', 'sentiment']

categorical_cols = ['Type', 'Name', 'Breed1', 'Gender', 'Color1', 'Color2',

       'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',

       'Sterilized', 'Health', 'Quantity', 'State', 'VideoAmt', 'PhotoAmt']
def create_data_bunch(use_val=True):

    if use_val:

        return TabularDataBunch.from_df(

            WORKING_PATH, 

            train_data[used_cols],

            dep_var = dependant_var,

            valid_idx = valid_id,

            cat_names = categorical_cols,

            cont_names = continious_cols,

            procs = procedures,

        )

    else:

        return TabularDataBunch.from_df(

            WORKING_PATH, 

            train_data[used_cols],

            dep_var = dependant_var,

            valid_idx = [],

            cat_names = categorical_cols,

            cont_names = continious_cols,

            procs = procedures,

        )



data_bunch = create_data_bunch()
def create_tabular_learner(data_bunch=data_bunch):

    learner = tabular_learner(

        data_bunch,

        layers=[200,100],

        emb_szs={

            'Name': 10,

            'Breed1':10,

        },

        metrics=accuracy

    )

    

    return learner



learner = create_tabular_learner()
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(1, 0.01)
from fastai.vision import *
#  Creating black images for pets without images



def generate_black_img(location):

    img_arr = np.zeros([224, 224, 3], dtype=np.uint8)

    img_arr.fill(0)

    imsave(location/"black_img.jpg", img_arr)



def add_photo_name_col(df):

    df_with_photos = df[df['PhotoAmt'] > 0].copy()

    df_with_photos['photo_name'] = df_with_photos['PetID'] + '-1.jpg'

    

#     df_with_photos = df.copy()

#     df['photo_name'] = df.progress_apply(lambda x: x['PetId'] + '-1.jpg' if x['PhotoAmt'] > 0 else "black_img.jpg", axis='columns')

    return df_with_photos





# generate_black_img(TRAIN_IMAGE_PATH)

image_train_data = add_photo_name_col(train_data)
train_image_data = ImageDataBunch.from_df(

    TRAIN_IMAGE_PATH,

    image_train_data[['photo_name', 'AdoptionSpeed']][:4000], 

    seed=1,

    ds_tfms=get_transforms(),

    size=112,

)



train_image_data.show_batch()
image_learner = cnn_learner(

    train_image_data,

    base_arch=models.resnet50,

    model_dir="/tmp/model/",

)



image_learner.lr_find()

image_learner.recorder.plot()
image_learner.fit_one_cycle(6, 0.01)

image_learner.save("112_6cycles")
image_learner.unfreeze()



image_learner.lr_find()

image_learner.recorder.plot()
image_learner.load("112_6cycles")

image_learner.fit_one_cycle(3, slice(1e-5, 0.01))

image_learner.save("112_6cycles_2cycles")
image_learner.freeze()



train_image_data = ImageDataBunch.from_df(

    TRAIN_IMAGE_PATH,

    image_train_data[['photo_name', 'AdoptionSpeed']], 

    seed=1,

    ds_tfms=get_transforms(),

    size=224,

)



image_learner.data = train_image_data



image_learner.lr_find()

image_learner.recorder.plot()
image_learner.fit_one_cycle(5, 1e-3)

image_learner.save("224_5cycles")
image_learner.unfreeze()

image_learner.lr_find()

image_learner.recorder.plot()
image_learner.load("224_5cycles")

image_learner.fit_one_cycle(3, slice(1e-5,1e-3 / 3))
def predict_image(row, img_path, model):

    if row['PhotoAmt'] > 0:

        # Might be a good idea to take the average of all picture predictions

        return model.predict(open_image(img_path/(row['PetID'] + "-1.jpg")))[2].numpy()

    else:

        return model.predict(Image(torch.from_numpy(np.random.random([3, 224, 224])).float()))[2].numpy()



def add_image_preds(model, img_path, tab_data):

    copy_of_tab = tab_data.copy()

    

    img_preds = tab_data.progress_apply(lambda x: predict_image(x, img_path, model), axis='columns')

    column_names = [f"embedding_{i}" for i in range(img_preds.iloc[0].shape[0])]

    copy_of_tab[column_names] = pd.DataFrame(img_preds.values.tolist())

    return copy_of_tab

    
train_data_with_image = add_image_preds(image_learner, TRAIN_IMAGE_PATH, train_data)
# full_data = train_data.append(valid_data)

# full_data_bunch = create_data_bunch(use_val=False)



def create_data_bunch_emb(dataset, visual_learner, use_val=True):    

    used_cols = [col for col in dataset.columns if col not in not_used_cols]

    

    if 'embedding_0' not in dataset.columns:

        dataset = add_image_preds(visual_learner, TRAIN_IMAGE_PATH, dataset)

    

    data_with_image = TabularDataBunch.from_df(

        WORKING_PATH, 

        dataset[used_cols],

        dep_var = dependant_var,

        valid_idx = valid_id if use_val else [],

        cat_names = categorical_cols,

        cont_names = continious_cols,

        procs = procedures,

    )



    return data_with_image



full_data_bunch = create_data_bunch_emb(train_data_with_image, image_learner)

full_learner = create_tabular_learner(data_bunch=full_data_bunch)
full_learner.lr_find()

full_learner.recorder.plot()
full_learner = create_tabular_learner(data_bunch=full_data_bunch)

full_learner.fit_one_cycle(1, 0.005)
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat



def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings



def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Calculates the quadratic weighted kappa

    quadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return 1.0 - numerator / denominator
def get_prediction(learner, sample):

    return int(learner.predict(sample)[1])



def evaluate_model(learner, dataset, target='AdoptionSpeed'):

    dataset = dataset.reset_index()

    data_len = len(dataset)

    

    predictions = dataset.progress_apply(lambda x: get_prediction(learner, x), axis='columns')

#     predictions = Parallel(n_jobs=4)(delayed(get_prediction)(sample) for sample in tqdm(dataset.itertuples()))

    

    accuracy = ((dataset[target] == pd.Series(predictions)).sum() / data_len)

    qwk = quadratic_weighted_kappa(dataset[target], predictions, min_rating=0, max_rating=4)

    

    return accuracy, qwk, predictions
train_emb_data = add_image_preds(image_learner, TRAIN_IMAGE_PATH, train_data[:100])

valid_emb_data = add_image_preds(image_learner, TRAIN_IMAGE_PATH, valid_data[:2000])



train_acc, train_qwk, train_preds = evaluate_model(full_learner, train_emb_data[:100])

valid_acc, valid_qwk, valid_preds = evaluate_model(full_learner, valid_emb_data[:2000])



print(f"Training acc: {train_acc} | Validation acc: {valid_acc}")

print(f"Training qwk: {train_qwk:.03f} | Validation qwk: {valid_qwk:.03f}")
def prepare_test_preds(test_data, learner):

    test_preds = test_data.progress_apply(lambda x: get_prediction(learner, x), axis='columns')

    test_preds.index = test_data['PetID']

    test_preds = test_preds.reset_index()

    test_preds.columns = ['PetID', 'AdoptionSpeed']

    return test_preds
test_data_emb = add_image_preds(image_learner, TEST_IMAGE_PATH, test_data[:100])

test_preds_df = prepare_test_preds(test_data_emb, full_learner)

test_preds_df.to_csv(WORKING_PATH/"submission.csv", index=False)