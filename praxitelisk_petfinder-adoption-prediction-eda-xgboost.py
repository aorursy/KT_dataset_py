# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use('fivethirtyeight')



import lightgbm as lgb

import xgboost as xgb



from wordcloud import WordCloud



# Any results you write to the current directory are saved as output.
# I am grateful for the help of author of this kernel for the main idea to load the dataset and save memory space!!

# https://www.kaggle.com/theoviel/load-the-totality-of-the-data



train_dtypes = {

        'PetID':                            'str',

        'AdoptionSpeed ':                   'int8',

        'Type':                             'category',

        'Name':                             'str',

        'Age':                              'int8',

        'Breed1':                           'category',

        'Breed2':                           'category',

        'Gender':                           'category',

        'Color1':                           'category',

        'Color2':                           'category',

        'Color3':                           'category',

        'MaturitySize':                     'float16',

        'FurLength':                        'int8',

        'Vaccinated':                       'category',

        'Dewormed':                         'category',

        'Sterilized':                       'category',

        'Health':                           'category',

        'Quantity':                         'uint16',

        'State':                            'category',

        'Fee':                              'float',

        'RescuerID':                        'category',

        'VideoAmt':                         'uint16',

        'PhotoAmt':                         'uint16',

        'Description ':                     'str'

        }



breeds_dtypes = {

        'BreedID':                          'category',

        'Type ':                            'category',

        'BreedName':                        'str'

        }



colors_dtypes = {

        'ColorID':                          'category',

        'ColorName':                        'str'

        }



states_dtypes = {

        'StateID':                          'category',

        'StateName':                        'str'

        }
breeds = pd.read_csv('../input/breed_labels.csv', dtype=breeds_dtypes)

colors = pd.read_csv('../input/color_labels.csv',  dtype=colors_dtypes)

states = pd.read_csv('../input/state_labels.csv', dtype=states_dtypes)



train = pd.read_csv('../input/train/train.csv', dtype=train_dtypes)

test = pd.read_csv('../input/test/test.csv', dtype=train_dtypes)





train['dataset_type'] = 'train'

test['dataset_type'] = 'test'

train_and_test = pd.concat([train, test])
train_and_test.head()
train.describe()
train.info()
train.columns
train.shape
train.isna().sum()
train_and_test.shape
train.AdoptionSpeed.value_counts()
train.AdoptionSpeed.value_counts() * 100 / train.shape[0]
train['AdoptionSpeed'].value_counts().sort_index().plot('bar');

plt.title('Adoption speed classes counts');
breeds["Breed1"] = breeds.BreedID

#breeds.drop("BreedID", axis="columns", inplace=True)

#breeds.drop("Type", axis="columns", inplace=True)



train_and_test_with_breeds = pd.merge(train_and_test, breeds[["Breed1", "BreedName"]], how= 'left',

                                      on="Breed1")



train_and_test_with_breeds["BreedName_1"] = train_and_test_with_breeds.BreedName

train_and_test_with_breeds.drop("BreedName", axis="columns", inplace=True)



breeds["Breed2"] = breeds.Breed1

train_and_test_with_breeds = pd.merge(train_and_test_with_breeds, breeds[["Breed2", "BreedName"]], how= 'left',

                                      on="Breed2")



train_and_test_with_breeds["BreedName_2"] = train_and_test_with_breeds.BreedName

train_and_test_with_breeds.drop("BreedName", axis="columns", inplace=True)

 

train_and_test_with_breeds.head(4)

#all_data_and_breeds = pd.merge(all_data, breeds, on="")
train_and_test_with_breeds.shape
colors["Color1"] = colors.ColorID

#breeds.drop("BreedID", axis="columns", inplace=True)

#breeds.drop("Type", axis="columns", inplace=True)



train_and_test_with_breeds_colors = pd.merge(train_and_test_with_breeds, colors[["Color1", "ColorName"]], 

                                             how= 'left', on="Color1")



train_and_test_with_breeds_colors["ColorName_1"] = train_and_test_with_breeds_colors.ColorName

train_and_test_with_breeds_colors.drop("ColorName", axis="columns", inplace=True)



colors["Color2"] = colors.ColorID

train_and_test_with_breeds_colors = pd.merge(train_and_test_with_breeds_colors, colors[["Color2", "ColorName"]], 

                                      how= 'left', on="Color2")



train_and_test_with_breeds_colors["ColorName_2"] = train_and_test_with_breeds_colors.ColorName

train_and_test_with_breeds_colors.drop("ColorName", axis="columns", inplace=True)



colors["Color3"] = colors.ColorID

train_and_test_with_breeds_colors = pd.merge(train_and_test_with_breeds_colors, colors[["Color3", "ColorName"]], 

                                      how= 'left', on="Color3")



train_and_test_with_breeds_colors["ColorName_3"] = train_and_test_with_breeds_colors.ColorName

train_and_test_with_breeds_colors.drop("ColorName", axis="columns", inplace=True)

 

train_and_test_with_breeds_colors.head(4)

#all_data_and_breeds = pd.merge(all_data, breeds, on="")
train_and_test_with_breeds_colors.shape
states["State"] = states.StateID

#breeds.drop("BreedID", axis="columns", inplace=True)

#breeds.drop("Type", axis="columns", inplace=True)



train_and_test_with_breeds_colors_states = pd.merge(train_and_test_with_breeds_colors, states[["State", "StateName"]], 

                                             how= 'left', on="State")



'''

train_and_test_with_breeds_colors_states["ColorName_1"] = train_and_test_with_breeds_colors.ColorName

train_and_test_with_breeds_colors.drop("ColorName", axis="columns", inplace=True)



colors["Color2"] = colors.ColorID

train_and_test_with_breeds_colors = pd.merge(train_and_test_with_breeds_colors, colors[["Color2", "ColorName"]], 

                                      how= 'left', on="Color2")



train_and_test_with_breeds_colors["ColorName_2"] = train_and_test_with_breeds_colors.ColorName

train_and_test_with_breeds_colors.drop("ColorName", axis="columns", inplace=True)



colors["Color3"] = colors.ColorID

train_and_test_with_breeds_colors = pd.merge(train_and_test_with_breeds_colors, colors[["Color3", "ColorName"]], 

                                      how= 'left', on="Color3")



train_and_test_with_breeds_colors["ColorName_3"] = train_and_test_with_breeds_colors.ColorName

train_and_test_with_breeds_colors.drop("ColorName", axis="columns", inplace=True)

'''

 

train_and_test_with_breeds_colors_states.head(4)

#all_data_and_breeds = pd.merge(all_data, breeds, on="")
train_and_test_with_breeds_colors_states.shape
train_and_test_with_breeds_colors_states.info()
categorical_columns = list(train_and_test_with_breeds_colors_states.loc[:, ((train_and_test_with_breeds_colors_states.dtypes =="category") | (train_and_test_with_breeds_colors_states.dtypes =="object"))].columns)

numerical_columns = list(train_and_test_with_breeds_colors_states.loc[:, ~((train_and_test_with_breeds_colors_states.dtypes =="category") | (train_and_test_with_breeds_colors_states.dtypes =="object"))].columns)



categorical_columns.remove("PetID")

categorical_columns.remove("RescuerID")

categorical_columns.remove("Description")

categorical_columns.remove("dataset_type")

categorical_columns.remove("BreedName_1")

categorical_columns.remove("BreedName_2")

categorical_columns.remove("ColorName_1")

categorical_columns.remove("ColorName_2")

categorical_columns.remove("ColorName_3")

categorical_columns.remove("Name")

categorical_columns.remove("StateName")



print(categorical_columns)
numerical_columns.remove("AdoptionSpeed")



print(numerical_columns)
def categorical_univariate_and_bivariate_stats(df, feature):

    

    train_sample = df

    

    if feature in train_sample.columns:

    

        print("Top 10 most occurred categories for the categorical feature", feature)

        print(train_sample[feature].value_counts().head(10))



        f, axes = plt.subplots(1, 2, figsize=(21, 10))



        train_sample[feature].value_counts().head(10).plot.bar(ax=axes[0], colormap="BrBG")



        #train_sample.groupby(["AdoptionSpeed", feature]).count()["PetID"].unstack(0).sort_values(by=1, axis=0, ascending=False).head(10).plot.bar(ax=axes[1], colormap="coolwarm")

        train_sample.groupby(["AdoptionSpeed", feature]).count()["PetID"].unstack(0).head(10).plot.bar(ax=axes[1], colormap="coolwarm")



        

        f.suptitle("Categorical feature: "+" Univariate and Bivariate plots against the target variable")

        

    else:

        print("This feature has been removed from dataset due to high NaN rate or highly unbalanced values")

        

        

def logistic_fit(df, feature):

    

    import warnings

    warnings.filterwarnings("ignore")

    

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import precision_score

    from sklearn.metrics import recall_score

    from sklearn.metrics import f1_score

    from sklearn.metrics import r2_score

    '''

    train_sample = df



    if feature in train_sample.columns:

        

        from sklearn.linear_model import LogisticRegression

        

        f, axes = plt.subplots(1, 2, figsize=(21, 10))



        # test if there is a logistic relationship between the feature1 and the target.

        print()

        print("Fitting a logistic regression model for the feature", feature,"against the target variable")

        

               

        mask = ~train_sample[feature].isnull() & ~train_sample["AdoptionSpeed"].isnull()



        logmodel = LogisticRegression(C=1e5, solver='lbfgs')

        

        if feature in categorical_columns:        

            logmodel.fit(train_sample[feature][mask].cat.codes.values.reshape(-1,1), train_sample["AdoptionSpeed"][mask])

            predictions = logmodel.predict(train_sample[feature][mask].cat.codes.values.reshape(-1,1))

        else:

            logmodel.fit(train_sample[feature][mask].values.reshape(-1,1), train_sample["AdoptionSpeed"][mask])

            predictions = logmodel.predict(train_sample[feature][mask].values.reshape(-1,1))



        from sklearn.metrics import classification_report

        print(classification_report(train_sample["AdoptionSpeed"][mask], predictions))

        print("")

        print("accuracy score:", accuracy_score(train_sample["AdoptionSpeed"][mask], predictions))

        print("F1 score:", accuracy_score(train_sample["AdoptionSpeed"][mask], predictions))

        #print("R^2 score:", r2_score(train_sample["HasDetections"][mask], predictions))



        import scikitplot as skplt

        skplt.metrics.plot_confusion_matrix(train_sample["AdoptionSpeed"][mask], predictions, normalize=False,

                                            title = "Confusion matrix for the feature: "+feature+" against the target variable after fitting a logistic regression model",

                                           figsize=(10,8), text_fontsize='medium', cmap="BrBG", ax = axes[0])

        

        

        # import statsmodels.api as sm

        # print()

        # est = sm.Logit(train_sample["HasDetections"][mask], train_sample[feature][mask].cat.codes.values.reshape(-1,1))

        # result1=est.fit()

        # print(result1.summary())

        if feature in categorical_columns:

            axes[1] = plt.scatter(train_sample[feature][mask].cat.codes.values.reshape(-1,1), predictions)

            axes[1] = plt.scatter(train_sample[feature][mask].cat.codes.values.reshape(-1,1), logmodel.predict_proba(train_sample[feature][mask].cat.codes.values.reshape(-1,1))[:,1])

            plt.xlabel(feature)

            plt.ylabel("HasDetections Probability")

            plt.title("Probability of Detecting a Malware vs the "+ feature)

            plt.show()

        else:

            axes[1] = plt.scatter(train_sample[feature][mask].values.reshape(-1,1), predictions)

            axes[1] = plt.scatter(train_sample[feature][mask].values.reshape(-1,1), logmodel.predict_proba(train_sample[feature][mask].values.reshape(-1,1))[:,1])

            plt.xlabel(feature)

            plt.ylabel("HasDetections Probability")

            plt.title("Probability of Detecting a Malware vs the "+ feature)

            plt.show()

        

    else:

        print("This feature has been removed from dataset due to high NaN rate or highly unbalanced values")

    '''

    from sklearn.tree import DecisionTreeClassifier



    train_df = df



    clf = DecisionTreeClassifier()

    

    if ((feature == "State") or (feature in numerical_columns)):

        clf.fit(train_df[feature].values.reshape(-1, 1), train_df["AdoptionSpeed"])

        predictions = clf.predict(train_df[feature].values.reshape(-1, 1))

    else:

        clf.fit(train_df[feature].cat.codes.values.reshape(-1, 1), train_df["AdoptionSpeed"])

        predictions = clf.predict(train_df[feature].cat.codes.values.reshape(-1, 1))



    from sklearn.metrics import accuracy_score

    from sklearn.metrics import precision_score

    from sklearn.metrics import recall_score

    from sklearn.metrics import f1_score

    from sklearn.metrics import r2_score



    from sklearn.metrics import classification_report

    print(classification_report(train_df["AdoptionSpeed"], predictions))



    import scikitplot as skplt

    skplt.metrics.plot_confusion_matrix(train_df["AdoptionSpeed"], predictions, normalize=False,

                                        figsize=(10,8), text_fontsize='medium')



#print("F1 score:", accuracy_score( train_df["AdoptionSpeed"], predictions))
train_df = train_and_test_with_breeds_colors_states[train_and_test_with_breeds_colors_states.dataset_type == "train"]



categorical_univariate_and_bivariate_stats(train_df, feature="Name")

#logistic_fit(train_df, feature="Type")
train_df = train_and_test_with_breeds_colors_states[train_and_test_with_breeds_colors_states.dataset_type == "train"]



categorical_univariate_and_bivariate_stats(train_df, feature="Type")

logistic_fit(train_df, feature="Type")
categorical_univariate_and_bivariate_stats(train_df, feature="Vaccinated")

logistic_fit(train_df, feature="Vaccinated")
categorical_univariate_and_bivariate_stats(train_df, feature="Sterilized")

logistic_fit(train_df, feature="Sterilized")
categorical_univariate_and_bivariate_stats(train_df, feature="Dewormed")

logistic_fit(train_df, feature="Dewormed")
categorical_univariate_and_bivariate_stats(train_df, feature="State")

logistic_fit(train_df, feature="State")
categorical_univariate_and_bivariate_stats(train_df, feature="Color1")

logistic_fit(train_df, feature="Color1")
import pandas as pd

import numpy as np

import scipy.stats as stats

from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction and has p-value {1}".format(colX, self.p)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)



cT = ChiSquare(train_df)        

for var in categorical_columns:

    cT.TestIndependence(colX=var,colY="AdoptionSpeed" )
from statsmodels.graphics.mosaicplot import mosaic

import matplotlib.pyplot as plt

import pandas



sns.set(rc={'figure.figsize':(13, 8)})

#mosaic(train_df, ['AdoptionSpeed', 'Type'])

#plt.show()



tab = pd.crosstab(train_df['AdoptionSpeed'], train_df['Type'])

mosaic(tab.stack(), title="Mosaic Plot")

plt.show()
def numerical_univariate_and_bivariate_plot(df, feature, num_of_bins = 40):

    

    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    

    train_sample = df

    

    if feature in train_sample.columns:

    

        print("Top 10 Values counts for the numerical feature", feature)

        print(train_sample[feature].value_counts().head(10))

        print("Min value", train_sample[feature].min())

        print("Max value", train_sample[feature].max())

        print("NaN values", train_sample[feature].isnull().sum())

        print("Number of unique values", train_sample[feature].nunique())



        if train[feature].nunique() > 2:

            print("Mean value", train_sample[feature].mean())

            print("Variance value", train_sample[feature].var())



        # for binary features

        if train[feature].nunique() <= 2:



            f, axes = plt.subplots(1, 2, figsize=(21, 10))



            sns.countplot(x=feature, data=train_sample, ax=axes[0])

            sns.countplot(x=feature, hue = "AdoptionSpeed", data=train_sample, ax=axes[1])

            

            f.suptitle("Numerical feature: "+feature+" Univariate and Bivariate plots against the target variable")



        # for numeric features

        else:



            f, axes = plt.subplots(1, 3, figsize=(21, 10))



            sns.distplot(train_sample[feature].dropna(), rug=False, kde=False, ax=axes[0], bins = num_of_bins)



            #sns.violinplot(x="AdoptionSpeed", y = feature, hue="AdoptionSpeed", data=train_sample, ax=axes[1])

            sns.boxplot(x="AdoptionSpeed", y = feature, hue="AdoptionSpeed", data=train_sample, ax=axes[1])



            if feature == "LocaleEnglishNameIdentifier":

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 0][feature].dropna().astype("int16"), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 1][feature].dropna().astype("int16"), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

            else:

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 0][feature].dropna(), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 1][feature].dropna(), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 2][feature].dropna(), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 3][feature].dropna(), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

                sns.distplot(train_sample[train_sample["AdoptionSpeed"] == 4][feature].dropna(), rug=False, kde=False, ax=axes[2], bins = num_of_bins)

            

                f.suptitle("Numerical feature: "+feature+" Univariate and Bivariate plots against the target variable")

    else:

        print("This feature has been removed from dataset due to high NaN rate or highly unbalanced values")
numerical_univariate_and_bivariate_plot(train_df, feature="Age")

logistic_fit(train_df, "Age")
numerical_univariate_and_bivariate_plot(train_df, feature="Fee")

logistic_fit(train_df, "Fee")
numerical_univariate_and_bivariate_plot(train_df, feature="FurLength")

logistic_fit(train_df, "FurLength")
numerical_univariate_and_bivariate_plot(train_df, feature="MaturitySize")

logistic_fit(train_df, "MaturitySize")
numerical_univariate_and_bivariate_plot(train_df, feature="PhotoAmt")

logistic_fit(train_df, "PhotoAmt")
numerical_univariate_and_bivariate_plot(train_df, feature="Quantity")

logistic_fit(train_df, "Quantity")
numerical_univariate_and_bivariate_plot(train_df, feature="VideoAmt")

logistic_fit(train_df, "VideoAmt")
sns.set(rc={'figure.figsize':(10, 8)})



# Compute the correlation matrix

corr = train_df[numerical_columns].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')
# I want to thank https://www.kaggle.com/artgor/exploration-of-data-step-by-step/notebook for the following snippet:



fig, ax = plt.subplots(figsize = (16, 12))

plt.subplot(1, 2, 1)

text_dog = ' '.join(train_and_test.loc[train_and_test['Type'] == '1', 'Name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='Black',

                      width=1200, height=1000).generate(text_dog)

plt.imshow(wordcloud)

plt.title('Top dog names')

plt.axis("off")



plt.subplot(1, 2, 2)

text_cat = ' '.join(train_and_test.loc[train_and_test['Type'] == '2', 'Name'].fillna('').values)

wordcloud = WordCloud(max_font_size=None, background_color='Black',

                      width=1200, height=1000).generate(text_cat)

plt.imshow(wordcloud)

plt.title('Top cat names')

plt.axis("off")



plt.show()
train_df = train_and_test_with_breeds_colors_states[train_and_test_with_breeds_colors_states.dataset_type == "train"]

test_df = train_and_test_with_breeds_colors_states[train_and_test_with_breeds_colors_states.dataset_type == "test"]
train_df.columns
test_df.columns
# A big thank you to https://www.kaggle.com/econdata/petfinder-lgbm/notebook for his intuition



%time

import json



train_id = train['PetID']

test_id = test['PetID']

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



train_df['doc_sent_mag'] = doc_sent_mag

train_df['doc_sent_score'] = doc_sent_score



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



test_df['doc_sent_mag'] = doc_sent_mag

test_df['doc_sent_score'] = doc_sent_score
%%time

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



train_desc = train.Description.fillna("none").values

test_desc = test.Description.fillna("none").values





max_train_len = [len(x) for x in train_desc]



max_test_len = [len(x) for x in test_desc]





tfv = TfidfVectorizer(min_df=3,  max_features=max([max(max_train_len), max(max_test_len)]),

        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',

        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,

        stop_words = 'english')

    

# Fit TFIDF

tfv.fit(list(train_desc))

X =  tfv.transform(train_desc)

X_test = tfv.transform(test_desc)



components = 480

svd = TruncatedSVD(n_components=components)

svd.fit(X)

print(svd.explained_variance_ratio_.sum())

print(svd.explained_variance_ratio_)

X = svd.transform(X)

X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(components)])

train_df = pd.concat((train_df, X), axis=1)

X_test = svd.transform(X_test)

X_test = pd.DataFrame(X_test, columns=['svd_{}'.format(i) for i in range(components)])



test_df.reset_index(drop=True, inplace=True)

X_test.reset_index(drop=True, inplace=True)



test_df = pd.concat([test_df, X_test], axis=1)
# I want to thank https://www.kaggle.com/econdata/petfinder-lgbm/notebook for his intuition



%time

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

train_df.loc[:, 'vertex_x'] = vertex_xs

train_df.loc[:, 'vertex_y'] = vertex_ys

train_df.loc[:, 'bounding_confidence'] = bounding_confidences

train_df.loc[:, 'bounding_importance'] = bounding_importance_fracs

train_df.loc[:, 'dominant_blue'] = dominant_blues

train_df.loc[:, 'dominant_green'] = dominant_greens

train_df.loc[:, 'dominant_red'] = dominant_reds

train_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

train_df.loc[:, 'dominant_score'] = dominant_scores

train_df.loc[:, 'label_description'] = label_descriptions

train_df.loc[:, 'label_score'] = label_scores





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

test_df.loc[:, 'vertex_x'] = vertex_xs

test_df.loc[:, 'vertex_y'] = vertex_ys

test_df.loc[:, 'bounding_confidence'] = bounding_confidences

test_df.loc[:, 'bounding_importance'] = bounding_importance_fracs

test_df.loc[:, 'dominant_blue'] = dominant_blues

test_df.loc[:, 'dominant_green'] = dominant_greens

test_df.loc[:, 'dominant_red'] = dominant_reds

test_df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs

test_df.loc[:, 'dominant_score'] = dominant_scores

test_df.loc[:, 'label_description'] = label_descriptions

test_df.loc[:, 'label_score'] = label_scores
train_df["HasName"] = np.where(train_df["Name"].isnull(), 1, 0)

train_df["HasDescription"] = np.where(train_df["Description"].isnull(), 1, 0)



test_df["HasName"] = np.where(test_df["Name"].isnull(), 1, 0)

test_df["HasDescription"] = np.where(test_df["Description"].isnull(), 1, 0)



train_df.drop(["Name", "PetID", "RescuerID", "dataset_type", "BreedName_1", "BreedName_2", "ColorName_1", "ColorName_2", "ColorName_3",

                    "StateName", "Description"], axis="columns", inplace = True)



test_df.drop(["Name", "PetID", "RescuerID", "dataset_type", "BreedName_1", "BreedName_2", "ColorName_1", "ColorName_2", "ColorName_3",

                    "StateName", "Description"], axis="columns", inplace = True)
train_shape = train_df.shape

test_shape = test_df.shape



train_and_test = pd.concat([train_df, test_df], axis="rows", sort=False)

train_and_test.head()
train_and_test.tail()
#train_and_test.drop(["Name", "PetID", "RescuerID", "dataset_type", "BreedName_1", "BreedName_2", "ColorName_1", "ColorName_2", "ColorName_3",

#                    "StateName", "Description"], axis="columns", inplace = True)
categorical_columns.append("label_description")

#categorical_columns.remove("label_description")



categorical_columns
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder







def MultiLabelEncoder(columnlist,dataframe):

    for i in columnlist:

        #print(i)

        labelencoder_X=LabelEncoder()

        dataframe[i]=labelencoder_X.fit_transform(dataframe[i])



        

train_and_test.loc[:, categorical_columns] = train_and_test[categorical_columns].astype('category')

MultiLabelEncoder(categorical_columns, train_and_test)
train_df = train_and_test[0:train_shape[0]]

test_df = train_and_test[(train_shape[0]):(train_and_test.shape[0]+1)]
test_df.columns
train_df.columns
test_df = test_df.drop(["AdoptionSpeed"], axis = 1)
train_df['AdoptionSpeed'] = train_df['AdoptionSpeed'].astype("category")
y = train_df['AdoptionSpeed']

X = train_df.drop(['AdoptionSpeed'], axis=1)
def xgbooft_all_purpose(X, y, type_of_training):

    

    from sklearn.model_selection import train_test_split, StratifiedKFold

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import precision_score

    from sklearn.metrics import recall_score

    from sklearn.metrics import f1_score

    from sklearn.metrics import classification_report

    from sklearn.metrics import roc_auc_score

    import scikitplot as skplt

    import time

    import random

    

    import xgboost as xgb

    

    # xgboost parameters

    eta = 0.01

    estimators  = 8000

    depth = 8

    gamma_value = 0.4

    colsample_bytree_value = 0.6

    max_rounds = 400

    

    if type_of_training == "baseline":

    # create a 70/30 split of the data 

        xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)

    

        import xgboost as xgb



        start_time = time.time()



        clf_xgb = xgb.XGBClassifier(learning_rate=eta, 

                                    n_estimators=estimators, 

                                    max_depth=depth,

                                    min_child_weight=1,

                                    gamma=gamma_value,

                                    subsample=1,

                                    colsample_bytree=colsample_bytree_value,

                                    objective= 'multi:softmax',

                                    nthread=-1,

                                    scale_pos_weight=1,

                                    reg_alpha = 0,

                                    reg_lambda = 1,

                                    seed=42)



        clf_xgb.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xvalid, yvalid)], 

                    early_stopping_rounds=max_rounds, eval_metric='mlogloss', verbose=100)



        predictions = clf_xgb.predict(xvalid)

        predictions_probas = clf_xgb.predict_proba(xvalid)



        print()

        print(classification_report(yvalid, predictions))



        print()

        print("f1_score", f1_score(yvalid, predictions, average = "macro"))



        print()

        print("elapsed time in seconds: ", time.time() - start_time)

        

        skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)

        

        predictions_probas = clf_xgb.predict_proba(xvalid)

        skplt.metrics.plot_roc(yvalid, predictions_probas)

        

        skplt.metrics.plot_precision_recall(yvalid, predictions_probas)

        

        xgb.plot_importance(clf_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



        print()

        #gc.collect()

        

        return clf_xgb, predictions, predictions_probas

        

    elif type_of_training == "stratified":

        

        xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)

        

        predictions_probas_list = []

        index_fold = 0

        best_score = 1

        

        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state = 42)

        

        clf_stra_xgb = xgb.XGBClassifier(learning_rate=eta, 

                                    n_estimators=estimators, 

                                    max_depth=depth,

                                    min_child_weight=1,

                                    gamma=gamma_value,

                                    subsample=1,

                                    colsample_bytree=colsample_bytree_value,

                                    objective= 'multi:softmax',

                                    nthread=-1,

                                    scale_pos_weight=1,

                                    reg_alpha = 0,

                                    reg_lambda = 1,

                                    seed=42)

        

        for train_index, valid_index in folds.split(xtrain, ytrain):

            xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

            ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



            print("Stratified Fold:", index_fold)

            index_fold = index_fold + 1

            

            import xgboost as xgb



            start_time = time.time()





            clf_stra_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                        early_stopping_rounds=max_rounds, eval_metric='mlogloss', verbose=100)

            

            #if (clf_stra_xgb.best_score < best_score):

            #    clf_best_stra_xgb = clf_stra_xgb

            #    best_score = clf_stra_xgb.best_score

            

            print()



            predictions_probas = clf_stra_xgb.predict_proba(xvalid)

            predictions_probas_list.append(predictions_probas)

            

        

        predictions_probas=[sum(i)/index_fold for i in zip(*predictions_probas_list)]

        predictions = np.argmax(predictions_probas, axis=1)

        

        #xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)

        #clf_stra_xgb = clf_best_stra_xgb

        #del clf_best_stra_xgb

        #print("Best score:", best_score)

        

        predictions = clf_stra_xgb.predict(xvalid)

        predictions_probas = clf_stra_xgb.predict_proba(xvalid)



        print()

        print(classification_report(yvalid, predictions))



        print()

        print("f1_score", f1_score(yvalid, predictions, average = "macro"))



        print()

        print("elapsed time in seconds: ", time.time() - start_time)

        

        skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)

        

        skplt.metrics.plot_roc(yvalid, predictions_probas)

        

        skplt.metrics.plot_precision_recall(yvalid, predictions_probas)

        

        xgb.plot_importance(clf_stra_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



        print()

        #gc.collect()

        return clf_stra_xgb, predictions, predictions_probas



    elif type_of_training == "oversampling":

        

        #### resampling techniques:

        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler



        # create a 70/30 split of the data 

        xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



        # RandomOverSampler

        ros = RandomOverSampler(random_state=42)

        X_resampled, y_resampled = ros.fit_resample(xtrain, ytrain)

        xtrain=pd.DataFrame(X_resampled, columns = X.columns)

        ytrain = y_resampled

        



        start_time = time.time()



        clf_ros_xgb = xgb.XGBClassifier(learning_rate=eta, 

                                    n_estimators=estimators, 

                                    max_depth=depth,

                                    min_child_weight=1,

                                    gamma=gamma_value,

                                    subsample=1,

                                    colsample_bytree=colsample_bytree_value,

                                    objective= 'multi:softmax',

                                    nthread=-1,

                                    scale_pos_weight=1,

                                    reg_alpha = 0,

                                    reg_lambda = 1,

                                    seed=42)



        clf_ros_xgb.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xvalid, yvalid)], 

                    early_stopping_rounds=max_rounds, eval_metric='mlogloss', verbose=100)



        predictions = clf_ros_xgb.predict(xvalid)

        predictions_probas = clf_ros_xgb.predict_proba(xvalid)



        print()

        print(classification_report(yvalid, predictions))



        print()

        print("f1_score", f1_score(yvalid, predictions, average = "macro"))



        print()

        print("elapsed time in seconds: ", time.time() - start_time)

        

        skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)

        

        skplt.metrics.plot_roc(yvalid, predictions_probas)

        

        skplt.metrics.plot_precision_recall(yvalid, predictions_probas)

        

        xgb.plot_importance(clf_ros_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



        print()

        #gc.collect()

        return clf_ros_xgb, predictions, predictions_probas

    

    elif type_of_training == "smote":

        #### resampling techniques:

        from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler



        # create a 70/30 split of the data 

        xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



        # SMOTE

        smote = SMOTE(random_state=42)

        X_resampled, y_resampled = smote.fit_resample(xtrain, ytrain)

        xtrain=pd.DataFrame(X_resampled, columns = X.columns)

        ytrain = y_resampled



        start_time = time.time()



        clf_smote_xgb = xgb.XGBClassifier(learning_rate=eta, 

                                    n_estimators=estimators, 

                                    max_depth=depth,

                                    min_child_weight=1,

                                    gamma=gamma_value,

                                    subsample=1,

                                    colsample_bytree=colsample_bytree_value,

                                    objective= 'multi:softmax',

                                    nthread=-1,

                                    scale_pos_weight=1,

                                    reg_alpha = 0,

                                    reg_lambda = 1,

                                    seed=42)



        clf_smote_xgb.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain), (xvalid, yvalid)], 

                    early_stopping_rounds=max_rounds, eval_metric='mlogloss', verbose=100)



        predictions = clf_smote_xgb.predict(xvalid)

        predictions_probas = clf_smote_xgb.predict_proba(xvalid)



        print()

        print(classification_report(yvalid, predictions))



        print()

        print("f1_score", f1_score(yvalid, predictions, average = "macro"))



        print()

        print("elapsed time in seconds: ", time.time() - start_time)

        

        skplt.metrics.plot_confusion_matrix(yvalid, predictions, normalize=True)

        

        skplt.metrics.plot_roc(yvalid, predictions_probas)

        

        skplt.metrics.plot_precision_recall(yvalid, predictions_probas)

        

        xgb.plot_importance(clf_smote_xgb, title='Feature importance', xlabel='F score', ylabel='Features')



        print()

        #gc.collect()

        return clf_smote_xgb, predictions, predictions_probas

    

    else:

        print("Please specify for the argument 'type_of_training'one of the following parameters: (as-is,stratified, oversampling, smote)")
clf_xgb, predictions, predictions_probas = xgbooft_all_purpose(X,y, type_of_training ="baseline")
clf_strat_xgb, predictions, predictions_probas = xgbooft_all_purpose(X,y, type_of_training ="stratified")
clf_ros_xgb, predictions, predictions_probas = xgbooft_all_purpose(X,y, type_of_training ="oversampling")
clf_smote_xgb, predictions, predictions_probas = xgbooft_all_purpose(X,y, type_of_training ="smote")
predictions = clf_xgb.predict(test_df)

                   

submission = pd.read_csv('../input/test/sample_submission.csv')

submission['AdoptionSpeed'] = [int(i) for i in predictions]





submission.to_csv('submission.csv', index=False)