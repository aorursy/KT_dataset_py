from fastai.utils.collect_env import *

show_install()


import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

np.random.seed(42)

from fastai.tabular import *

from sklearn.model_selection import StratifiedKFold

seed=42

from fastai.callbacks.tracker import *
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.metrics import confusion_matrix as sk_cmatrix

from sklearn.model_selection import StratifiedKFold



import numpy as np





import scipy as sp



from collections import Counter

from functools import partial

from math import sqrt







# FROM: https://www.kaggle.com/myltykritik/simple-lgbm-image-features



# The following 3 functions have been taken from Ben Hamner's github repository

# https://github.com/benhamner/Metrics

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





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

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

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

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



    return (1.0 - numerator / denominator)



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4



        ll = quadratic_weighted_kappa(y, X_p)

        return -ll



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']

    

def rmse(actual, predicted):

    return sqrt(mean_squared_error(actual, predicted))
# def get_learner(data,layers,save_name='best_nn'):

#     return tabular_learner(data, layers=layers, metrics=[accuracy],

#                            callback_fns=[partial(SaveModelCallback, monitor='val_loss', name=save_name)]

#                           )

# def get_learner_no_cb(data,layers):

#     return tabular_learner(data, layers=layers, metrics=[accuracy,QuadraticKappaScore()])





def get_learner(data,layers,save_name='best_nn',y_range=None):

    return tabular_learner(data, layers=layers, 

                           callback_fns=[partial(SaveModelCallback, name=save_name)],

                           y_range=y_range

                          )

def get_learner_no_cb(data,layers,y_range=None):

    return tabular_learner(data, layers=layers, y_range=y_range)
def plot_pred(pred,target,alpha=0.1):

    fig,axes = plt.subplots(1,2,figsize=(12,4))

    axes[0].hist(pred,range=[0,5])

    axes[1].scatter(range(0,len(target)),target,label='True')

    axes[1].scatter(range(0,len(pred)),pred,c='r',alpha=alpha,label='Pred')

    axes[1].set_ylim(-1,5)

    axes[1].legend()

def evaluate_classification(preds,targets,coeff=None):

    optR = OptimizedRounder()

    if not coeff:

        optR.fit(preds, targets)

        coeff = optR.coefficients()

    pred_clas = optR.predict(preds, coeff)

    print(f"Pred dist\n{pd.value_counts(pred_clas,normalize=True).sort_index()}")

    print(f"True dist\n{pd.value_counts(targets,normalize=True).sort_index()}")

    print("Coefficients: ", coeff)

    qwk = quadratic_weighted_kappa(targets, pred_clas)

    print("QWK = ", qwk)

    return coeff
# class QuadraticKappaScore(ConfusionMatrix):

#     """

#     Compute the rate of agreement (Cohens Kappa).

#     Ref.: https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py

#     """

    

#     def on_epoch_end(self, **kwargs):

#         w = torch.zeros((self.n_classes, self.n_classes))

#         w = w + torch.arange(self.n_classes).to(dtype=torch.float32)

#         w = (w - w.transpose(0,1))**2

#         sum0 = self.cm.sum(dim=0)

#         sum1 = self.cm.sum(dim=1)

#         expected = torch.einsum('i,j->ij', (sum0, sum1)) / sum0.sum()

#         k = torch.sum(w * self.cm) / torch.sum(w * expected)

#         self.metric = 1 - k
print('Train')

train = pd.read_csv("../input/train/train.csv")

print(train.shape)



print('Test')

test = pd.read_csv("../input/test/test.csv")

print(test.shape)



print('Breeds')

breeds = pd.read_csv("../input/breed_labels.csv")

print(breeds.shape)



print('Colors')

colors = pd.read_csv("../input/color_labels.csv")

print(colors.shape)



print('States')

states = pd.read_csv("../input/state_labels.csv")

print(states.shape)
target = train['AdoptionSpeed']

train_id = train['PetID']

test_id = test['PetID']

train.drop(['AdoptionSpeed', 'PetID'], axis=1, inplace=True)

test.drop(['PetID'], axis=1, inplace=True)
doc_sent_mag = []

doc_sent_score = []

nf_count = 0

for pet in train_id:

    try:

        with open('../input/train_sentiment/' + pet + '.json', 'r') as f:

            sentiment = json.load(f)

        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])

        doc_sent_score.append(sentiment['documentSentiment']['score'])

    except FileNotFoundError:

        nf_count += 1

        doc_sent_mag.append(-1)

        doc_sent_score.append(-1)



train.loc[:, 'doc_sent_mag'] = doc_sent_mag

train.loc[:, 'doc_sent_score'] = doc_sent_score



doc_sent_mag = []

doc_sent_score = []

nf_count = 0

for pet in test_id:

    try:

        with open('../input/test_sentiment/' + pet + '.json', 'r') as f:

            sentiment = json.load(f)

        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])

        doc_sent_score.append(sentiment['documentSentiment']['score'])

    except FileNotFoundError:

        nf_count += 1

        doc_sent_mag.append(-1)

        doc_sent_score.append(-1)



test.loc[:, 'doc_sent_mag'] = doc_sent_mag

test.loc[:, 'doc_sent_score'] = doc_sent_score

train.Name = train.Name.str.lower()

test.Name = test.Name.str.lower()



train.Name.fillna("no name",inplace=True)

test.Name.fillna("no name",inplace=True)



train['name_length']= train.Name.str.len()

train.loc[train.Name.str.contains('no name|unknown|unname|noname|none'),'name_length']=0



test['name_length']= test.Name.str.len()

test.loc[test.Name.str.contains('no name|unknown|unname|noname|none'),'name_length']=0
train_desc = train.Description.fillna("none").values

test_desc = test.Description.fillna("none").values
tfv = TfidfVectorizer(min_df=3,  max_features=10000,

        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',

        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,

        stop_words = 'english')

    

# Fit TFIDF

tfv.fit(list(train_desc))

X =  tfv.transform(train_desc)

X_test = tfv.transform(test_desc)



svd = TruncatedSVD(n_components=120,random_state=42)

svd.fit(X)

# print(svd.explained_variance_ratio_.sum())

# print(svd.explained_variance_ratio_)

X = svd.transform(X)

print("X (svd):", X.shape)



X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(120)])

train = pd.concat((train, X), axis=1)

X_test = svd.transform(X_test)

X_test = pd.DataFrame(X_test, columns=['svd_{}'.format(i) for i in range(120)])

test = pd.concat((test, X_test), axis=1)
vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in train_id:

    try:

        with open('../input/train_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

print(nl_count)

train.loc[:, 'vertex_x'] = vertex_xs

train.loc[:, 'vertex_y'] = vertex_ys

train.loc[:, 'bounding_confidence'] = bounding_confidences

train.loc[:, 'bounding_importance'] = bounding_importance_fracs

train.loc[:, 'dominant_blue'] = dominant_blues

train.loc[:, 'dominant_green'] = dominant_greens

train.loc[:, 'dominant_red'] = dominant_reds

train.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

train.loc[:, 'dominant_score'] = dominant_scores

train.loc[:, 'label_description'] = label_descriptions

train.loc[:, 'label_score'] = label_scores





vertex_xs = []

vertex_ys = []

bounding_confidences = []

bounding_importance_fracs = []

dominant_blues = []

dominant_greens = []

dominant_reds = []

dominant_pixel_fracs = []

dominant_scores = []

label_descriptions = []

label_scores = []

nf_count = 0

nl_count = 0

for pet in test_id:

    try:

        with open('../input/test_metadata/' + pet + '-1.json', 'r') as f:

            data = json.load(f)

        vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']

        vertex_xs.append(vertex_x)

        vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']

        vertex_ys.append(vertex_y)

        bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']

        bounding_confidences.append(bounding_confidence)

        bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)

        bounding_importance_fracs.append(bounding_importance_frac)

        dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']

        dominant_blues.append(dominant_blue)

        dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']

        dominant_greens.append(dominant_green)

        dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']

        dominant_reds.append(dominant_red)

        dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']

        dominant_pixel_fracs.append(dominant_pixel_frac)

        dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']

        dominant_scores.append(dominant_score)

        if data.get('labelAnnotations'):

            label_description = data['labelAnnotations'][0]['description']

            label_descriptions.append(label_description)

            label_score = data['labelAnnotations'][0]['score']

            label_scores.append(label_score)

        else:

            nl_count += 1

            label_descriptions.append('nothing')

            label_scores.append(-1)

    except FileNotFoundError:

        nf_count += 1

        vertex_xs.append(-1)

        vertex_ys.append(-1)

        bounding_confidences.append(-1)

        bounding_importance_fracs.append(-1)

        dominant_blues.append(-1)

        dominant_greens.append(-1)

        dominant_reds.append(-1)

        dominant_pixel_fracs.append(-1)

        dominant_scores.append(-1)

        label_descriptions.append('nothing')

        label_scores.append(-1)



print(nf_count)

test.loc[:, 'vertex_x'] = vertex_xs

test.loc[:, 'vertex_y'] = vertex_ys

test.loc[:, 'bounding_confidence'] = bounding_confidences

test.loc[:, 'bounding_importance'] = bounding_importance_fracs

test.loc[:, 'dominant_blue'] = dominant_blues

test.loc[:, 'dominant_green'] = dominant_greens

test.loc[:, 'dominant_red'] = dominant_reds

test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

test.loc[:, 'dominant_score'] = dominant_scores

test.loc[:, 'label_description'] = label_descriptions

test.loc[:, 'label_score'] = label_scores
train.drop(['Name', 'RescuerID'], axis=1, inplace=True)

test.drop(['Name', 'RescuerID'], axis=1, inplace=True)
train.shape
train.drop(['Description'],inplace=True,axis=1)

test.drop(['Description'],inplace=True,axis=1)

target = target.astype(np.float32)

train['AdoptionSpeed'] = target
cont_names=['name_length','Age', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'doc_sent_mag', 'doc_sent_score', 'dominant_score', 'dominant_pixel_frac', 'dominant_red', 'dominant_green', 'dominant_blue', 'bounding_importance', 'bounding_confidence', 'vertex_x', 'vertex_y', 'label_score'] + ['svd_{}'.format(i) for i in range(120)]

cat_names = list(set(train.columns) - set(cont_names) - {'AdoptionSpeed'})

print(f'# of continuous feas: {len(cont_names)}')

print(f'# of categorical feas: {len(cat_names)}')

dep_var = 'AdoptionSpeed'

procs = [Categorify, Normalize]
train_idxs=[]

val_idxs=[]

skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

for train_idx, val_idx in skf.split(train, train.AdoptionSpeed):

    train_idxs.append(train_idx)

    val_idxs.append(val_idx)

val_idxs
def get_databunch(k=0,bs=64):

    data = (TabularList.from_df(train, path=Path("../"), cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(val_idxs[k])

                           .label_from_df(cols=dep_var)

                           .add_test(TabularList.from_df(test, path=Path("../"), cat_names=cat_names, cont_names=cont_names))

                           .databunch(bs=bs))

    return data
l2str= lambda layers: '_'.join([str(l) for l in layers])
# from fastai.utils.collect_env import *

# show_install()
from sklearn.metrics import cohen_kappa_score,accuracy_score



def metric(y1,y2):

    return cohen_kappa_score(y1,y2, weights='quadratic')
layers=[200,100]
data = get_databunch(0,bs=128)
model_name = f'best_{l2str(layers)}_reg_s'

model_name
learn = get_learner_no_cb(data,layers)

learn.lr_find()

learn.recorder.plot(skip_end=1)
# learn = get_learner(data,layers,model_name+'1')

# learn.fit_one_cycle(6,1e-01,pct_start=0.5)

# learn.recorder.plot_losses()
learn = get_learner(data,layers,model_name+'1')

learn.fit_one_cycle(6,1e-01,pct_start=0.5)

learn.recorder.plot_losses()
learn.load(model_name+'1')

val_preds=np.squeeze(to_np(learn.get_preds()[0]))

plot_pred(val_preds,data.valid_ds.y.items)
coeff=evaluate_classification(val_preds,data.valid_ds.y.items)
# coeff=[1.6,2.008,2.5,3.07]

# _=evaluate_classification(val_preds,data.valid_ds.y.items,coeff)
learn = get_learner_no_cb(data,layers)

learn.load(model_name+'1')

learn.lr_find()

learn.recorder.plot()
learn = get_learner(data,layers,model_name+'2')

learn.load(model_name+'1')

learn.fit_one_cycle(5,max_lr=3e-05)

learn.recorder.plot_losses()
# learn = get_learner(data,layers,model_name+'2')

# learn.load(model_name+'1')

# learn.fit_one_cycle(5,max_lr=3e-05)

# learn.recorder.plot_losses()
learn.load(model_name+'2')

val_preds=np.squeeze(to_np(learn.get_preds()[0]))

plot_pred(val_preds,data.valid_ds.y.items)
coeff = evaluate_classification(val_preds,data.valid_ds.y.items)

# TODO: need a better way to find these numbers

new_coeff=[1.65,coeff[1]+0.01,coeff[2],coeff[3]-0.01]

_=evaluate_classification(val_preds,data.valid_ds.y.items,new_coeff)
# new_coeff=[1.5,coeff[1]+0.16,coeff[2],coeff[3]-0.06]

# _=evaluate_classification(val_preds,data.valid_ds.y.items,new_coeff)
test_preds = np.squeeze(to_np(learn.get_preds(DatasetType.Test)[0]))
test_preds = OptimizedRounder().predict(test_preds,new_coeff)

test_preds=test_preds.astype(np.int32)

# test_preds
# Store predictions for Kaggle Submission

submission_df = pd.DataFrame(data={'PetID' : pd.read_csv('../input/test/test.csv')['PetID'], 

                                   'AdoptionSpeed' : test_preds})

submission_df.to_csv('submission.csv', index=False)