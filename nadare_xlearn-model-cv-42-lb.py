import os

import json

from operator import itemgetter

from collections import defaultdict

from functools import partial

from math import isnan



import cv2

import dill

import numpy as np

import pandas as pd

import scipy as sp

from tqdm import tqdm

from joblib import Parallel, delayed



from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import LabelEncoder
# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76106



os.mkdir("./working")

os.environ['USER'] = 'root'

os.system('pip install ../input/xlearn/xlearn/xlearn-0.40a1/')



import xlearn as xl

train_df = pd.read_csv("../input/petfinder-adoption-prediction/train/train.csv", encoding="utf-8")

test_df = pd.read_csv("../input/petfinder-adoption-prediction/test/test.csv", encoding="utf-8")



smpsb_df = pd.read_csv("../input/petfinder-adoption-prediction/test/sample_submission.csv")
train_df["group"] = 0

petid_map = {v: i for i, v in enumerate(pd.concat([train_df["PetID"], test_df["PetID"]]))}

rescuerid_encoder = LabelEncoder().fit(pd.concat([train_df["RescuerID"], test_df["RescuerID"]]))



for group, (_, group_idx) in enumerate(GroupKFold(n_splits=10).split(train_df,

                                                                     train_df["AdoptionSpeed"],

                                                                     rescuerid_encoder.transform(train_df["RescuerID"]))):

    train_df.loc[group_idx, "group"] = group
train_df.pivot_table(columns="AdoptionSpeed",

                     index="group",

                     values="Breed1",

                     aggfunc="count")
class FFMformatter():

    def __init__(self,

                 numerical_field_max: int,

                 key_column="PetID"

                 ):

        

        self.numerical_field_max = numerical_field_max

        self.key_column = key_column



        self.numerical_field_last = -1

        self.numerical_field = defaultdict(lambda: self.new_numerical_field())

        self.field_nan = defaultdict(lambda: np.NaN)

        

        self.categorical_field_last = numerical_field_max

        self.categorical_feature_last = numerical_field_max

        self.categorical_field = defaultdict(lambda: self.new_categorical_field())

        self.categorical_feature = defaultdict(lambda: defaultdict(lambda: self.new_categorical_feature()))

        self.result = {}

        self.categorical_columns = set()





    def add_dataframe(self, df: pd.DataFrame):

        petids = df[self.key_column]

        for field in df.columns:

            if field == self.key_column:

                continue

            for petid, value in zip(petids, df[field]):

                if self.field_nan[field] == value or ((type(value) == float) and isnan(value)):

                    continue



                if field in self.categorical_columns:

                    field_id = self.categorical_field[field]

                    self.result[petid] += " {}:{}:1".format(field_id,

                                                            self.categorical_feature[field_id][value])

                

                else:

                    field_id = self.numerical_field[field]

                    self.result[petid] += " {}:{}:{}".format(field_id,

                                                             field_id,

                                                             value)



    def add_field_dataframe(self, df, field_col, value_col):

        field_id = self.categorical_field[field_col]

        for _, row in tqdm(df.iterrows()):

            petid = row[self.key_column]

            self.result[petid] += " {}:{}:{}".format(field_id,

                                                     self.categorical_feature[field_id][row[field_col]],

                                                     row[value_col])

        

    def add_dataframe_as_samefield(self, df, basecol=None):

        assert self.key_column in df.columns

        cols = [col for col in df.columns if col != self.key_column]

        if basecol is None:

            basecol = cols[0]

        field_id = self.categorical_field[basecol]

        key_vals = df[self.key_column].values

        for col in cols:

            for key_val, value in zip(key_vals, df[col].values):

                if self.field_nan[basecol] == value or ((type(value) == float) and isnan(value)):

                    continue

                self.result[key_val] += " {}:{}:{}".format(field_id,

                                                           self.categorical_feature[field_id][col],

                                                           value)





    def set_categorical_columns(self, columns):

        if type(columns) == str:

            self.categorical_columns.add(columns)

        else:

            self.categorical_columns.update(columns)





    def set_multicolumns_as_column(self, columns):

        self.set_categorical_columns(columns)

        base_field = self.categorical_field[columns[0]]

        for col in columns:

            self.categorical_field[col] = base_field





    def add_Petids(self, keys, targets=None):

        if targets is not None:

            for key, target in zip(keys, targets):

                self.result[key] = str(target)

        else:

            for key in keys:

                self.result[key] = "-1"



    def set_field_nanvalue(self, col, nanvalue=np.NaN):

        self.field_nan[col] = nanvalue



    def new_numerical_field(self):

        self.numerical_field_last += 1

        if self.numerical_field_last > self.numerical_field_max:

            raise

        return self.numerical_field_last * 1



    def new_categorical_field(self):

        self.categorical_field_last += 1

        return self.categorical_field_last * 1

    

    def new_categorical_feature(self):

        self.categorical_feature_last += 1

        return self.categorical_feature_last * 1
df2ffm = FFMformatter(numerical_field_max=200)



df2ffm.add_Petids(train_df["PetID"].values, train_df["AdoptionSpeed"].values)

df2ffm.add_Petids(test_df["PetID"].values)
train_df["care_count"] = (train_df["Vaccinated"] == 1).astype(np.uint8) + (train_df["Dewormed"] == 1) + (train_df["Sterilized"] == 1)

test_df["care_count"] = (test_df["Vaccinated"] == 1).astype(np.uint8) + (test_df["Dewormed"] == 1) + (test_df["Sterilized"] == 1)



train_df["care_uncertain_count"] = ((train_df["Vaccinated"] == 3).astype(np.uint8) +

                                    (train_df["Dewormed"] == 3) +

                                    (train_df["Sterilized"] == 3))

test_df["care_uncertain_count"] = ((test_df["Vaccinated"] == 3).astype(np.uint8) +

                                   (test_df["Dewormed"] == 3) +

                                   (test_df["Sterilized"] == 3))



train_df["color_counts"] = 3 - (train_df.filter(regex="^Color") == 0).sum(axis=1)

test_df["color_counts"] = 3 - (test_df.filter(regex="^Color") == 0).sum(axis=1)



rescuer_count_map = pd.concat([train_df["RescuerID"], test_df["RescuerID"]]).value_counts().to_dict()

train_df["Rescuer_count"] = np.log(train_df["RescuerID"].map(rescuer_count_map))

test_df["Rescuer_count"] = np.log(test_df["RescuerID"].map(rescuer_count_map))



train_df["Fee"] = np.log1p(train_df["Fee"])

test_df["Fee"] = np.log1p(test_df["Fee"])



train_df["Age"] = np.log1p(train_df["Age"])

test_df["Age"] = np.log1p(test_df["Age"])



train_df["description_len"] = np.log1p(train_df["Description"].fillna("").str.len())

test_df["description_len"] = np.log1p(test_df["Description"].fillna("").str.len())



train_df["name_len"] = np.log1p(train_df["Name"].fillna("").str.len())

test_df["name_len"] = np.log1p(test_df["Name"].fillna("").str.len())



# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/78040



# state GDP: https://en.wikipedia.org/wiki/List_of_Malaysian_states_by_GDP

state_gdp = {

    41336: 116.679,

    41325: 40.596,

    41367: 23.02,

    41401: 190.075,

    41415: 5.984,

    41324: 37.274,

    41332: 42.389,

    41335: 52.452,

    41330: 67.629,

    41380: 5.642,

    41327: 81.284,

    41345: 80.167,

    41342: 121.414,

    41326: 280.698,

    41361: 32.270

}



# state population: https://en.wikipedia.org/wiki/Malaysia

state_population = {

    41336: 33.48283,

    41325: 19.47651,

    41367: 15.39601,

    41401: 16.74621,

    41415: 0.86908,

    41324: 8.21110,

    41332: 10.21064,

    41335: 15.00817,

    41330: 23.52743,

    41380: 2.31541,

    41327: 15.61383,

    41345: 32.06742,

    41342: 24.71140,

    41326: 54.62141,

    41361: 10.35977

}



train_df["state_gdp"] = train_df.State.map(state_gdp)

test_df["state_gdp"] = test_df.State.map(state_gdp)



train_df["state_population"] = train_df.State.map(state_population)

test_df["state_population"] = test_df.State.map(state_population)



for col in ["description_len", "name_len", "state_gdp", "state_population", "Rescuer_count", "Fee", "Age"]:

    mean, std = train_df[col].mean(), train_df[col].std()

    train_df[col] = (train_df[col] - mean)/std

    test_df[col] = (test_df[col] - mean)/std
train_df.head()
df2ffm.set_multicolumns_as_column(["Color1", "Color2", "Color3"])

df2ffm.set_multicolumns_as_column(["Breed1", "Breed2"])

df2ffm.set_categorical_columns(["Gender", "State"])



for col in ["Color1", "Color2", "Color3", "Breed1", "Breed2", "FurLength", "MaturitySize", "Health", "State"]:

    df2ffm.set_field_nanvalue(col, 0)



for col in ["Vacciated", "Dewormed", "Sterilized"]:

    df2ffm.set_field_nanvalue(col, 3)



drop_col = ["Name", "Description", "RescuerID"]

df2ffm.add_dataframe(train_df.drop(drop_col + ["AdoptionSpeed", "group"], axis=1))

df2ffm.add_dataframe(test_df.drop(drop_col, axis=1))
def load_metadata(path):

    file = path.split("/")[-1]

    pet_id = file[:-5].split("-")[0]

    file_id = file[:-5].split("-")[1]

    

    with open(path, encoding="utf-8") as f:

        jfile = json.loads(f.read())

    response = {"labels": [],

                "text": {"PetID": pet_id,

                         "FileID": file_id,

                         "description": ""}}

    

    if "labelAnnotations" in jfile.keys():

        for anot in jfile["labelAnnotations"]:

            response["labels"].append({"PetID": pet_id,

                                       "FileID": file_id,

                                       "description": anot["description"],

                                       "score": anot["score"]})



    if "imagePropertiesAnnotation" in jfile.keys():

        colors = np.zeros((10, 1, 3), dtype=np.uint8)

        scores = np.zeros(10)

        fractions = np.zeros(10)

        getscore = itemgetter("score")

        for i, color in enumerate(sorted(jfile['imagePropertiesAnnotation']["dominantColors"]["colors"],

                                         key=getscore,

                                         reverse=True)

                                 ):



            for j, c in enumerate(["red", "green", "blue"]):

                if not color["color"].get(c) is None:

                    colors[i, 0, j] = color["color"][c] 

                

            scores[i] = color["score"]

            fractions[i] = color["pixelFraction"]

        hsv = cv2.cvtColor(colors, cv2.COLOR_RGB2HSV_FULL)

        response["property"] = {"PetID": pet_id,

                                "FileID": file_id,

                                "top_red": colors[0, 0, 0],

                                "top_green": colors[0, 0, 1],

                                "top_blue": colors[0, 0, 2],

                                "top_score": scores[0],

                                "top_fraction": fractions[0],

                                "top_hue": hsv[0, 0, 0],

                                "top_saturation": hsv[0, 0, 1],

                                "top_brightness": hsv[0, 0, 2],

                                "top3_score": scores[:3].sum(),

                                "top3_fraction": fractions[:3].sum(),

                                "top3_area": np.linalg.norm(np.cross((colors[1] - colors[0])[0], (colors[2] - colors[0])[0])),

                                "top10_fraction": fractions.sum(),

                                "top10_score": scores.sum()}



    if 'cropHintsAnnotation' in jfile.keys():

        tmp = jfile["cropHintsAnnotation"]["cropHints"][0]

        response["crop"] = {"PetID": pet_id,

                            "FileID": file_id,

                            "confidence": tmp["confidence"]}

        if not tmp.get("importanceFraction") is None:

            response["crop"]["importanceFraction"] = tmp["importanceFraction"]

    

    if 'textAnnotations' in jfile.keys():

        for anot in jfile["textAnnotations"]:

            response["text"]["description"] += anot["description"] + " "

    

    if "faceAnnotations" in jfile.keys():

        faceanot = jfile["faceAnnotations"][0]

        response["face"] = {"PetID": pet_id,

                            "FileID": file_id,

                            "detectionConfidence": faceanot['detectionConfidence'],

                            'landmarkingConfidence': faceanot['landmarkingConfidence'],

                            }

    

    return response
metadata_path = [dir_ + file for dir_ in ["../input/petfinder-adoption-prediction/train_metadata/",

                                          "../input/petfinder-adoption-prediction/test_metadata/"]

                                 for file in os.listdir(dir_)]

results = Parallel(n_jobs=-1, verbose=50)([delayed(load_metadata)(path) for path in metadata_path])



labels = []

properties = []

crops = []

faces = []

texts = []

for res in tqdm(results):

    if not res.get("labels") is None:

        labels.extend(res["labels"])

    if not res.get("property") is None:

        properties.append(res["property"])

    if not res.get("crop") is None:

        crops.append(res["crop"])

    if not res.get("face") is None:

        faces.append(res["face"])

    if not res.get("text") is None:

        texts.append(res["text"])



labels_df = pd.DataFrame(labels)

properties_df = pd.DataFrame(properties)

crops_df = pd.DataFrame(crops)

faces_df = pd.DataFrame(faces)

texts_df = pd.DataFrame(texts)
labels_agg = labels_df.groupby(["PetID", "description"])["score"].max().reset_index()

df2ffm.add_field_dataframe(labels_agg.rename(columns={"description": "labels_description"}).reset_index(), "labels_description", "score")
ffm_df = pd.Series(df2ffm.result).reset_index()

ffm_df.columns = ["PetID", "ffm_text"]



train_ffm = train_df[["PetID", "AdoptionSpeed", "group"]].merge(ffm_df,

                                                                on="PetID",

                                                                how="left")



test_ffm = test_df[["PetID"]].merge(ffm_df,

                                    on="PetID",

                                    how="left")
with open("./working/test.txt", "w") as f:

    f.write("\n".join(test_ffm.loc[:, "ffm_text"].values.tolist()))

train_oof = np.zeros(len(train_df))

test_pred = np.zeros(len(test_df))



for j in tqdm(range(50)):

    i = j%10

    with open("./working/dev.txt", "w") as f:

        f.write("\n".join(train_ffm.loc[train_ffm["group"] != i, "ffm_text"].values.tolist()))

    with open("./working/val.txt", "w") as f:

        f.write("\n".join(train_ffm.loc[train_ffm["group"] == i, "ffm_text"].values.tolist()))

    param = {"task": "reg",

             "lr": .1,

             "epoch": 200,

             "lambda": .0001,

             "k": 4,

             "nthread": 4,

             "metric": "rmse"}

    ffm_model =xl.create_ffm()

    ffm_model.setTrain("./working/dev.txt")

    ffm_model.setValidate("./working/val.txt")

    ffm_model.fit(param, "./working/model.out")



    ffm_model.setTest("./working/val.txt")

    ffm_model.predict("./working/model.out", "./working/output.txt")

    output = pd.read_csv("./working/output.txt", header=None)[0].values

    train_oof[np.where(train_ffm["group"] == i)] += output / 5



    ffm_model.setTest("./working/test.txt")

    ffm_model.predict("./working/model.out", "./working/output.txt")

    output = pd.read_csv("./working/output.txt", header=None)[0].values

    test_pred += output / 50
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(train_df["AdoptionSpeed"], train_oof))
from sklearn.metrics import cohen_kappa_score



# https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights = 'quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = np.percentile(X, [2.73, 23.3, 50.3, 72]) # <= improved

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']





# https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76106

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
y_train = train_df["AdoptionSpeed"].values

group = train_df["group"].values



coef_mean = np.zeros(4)

for fold in range(10):

    dev_idx = np.where(group != fold)[0]

    val_idx = np.where(group == fold)[0]



    pred_dev = train_oof[dev_idx]

    y_dev = y_train[dev_idx]



    pred_val = train_oof[val_idx]

    y_val = y_train[val_idx]



    optR = OptimizedRounder()

    optR.fit(pred_dev, y_dev)

    coefficients = optR.coefficients()

    coef_mean += coefficients / 10

    pred_val_k = optR.predict(pred_val, coefficients)
quadratic_weighted_kappa(y_train, optR.predict(train_oof, coef_mean))
smpsb_df["AdoptionSpeed"] = optR.predict(test_pred, coef_mean)

smpsb_df.to_csv("submission.csv", index=None)