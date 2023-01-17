import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import scipy.stats as stats

from scipy.stats import randint



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



pd.set_option('display.max_columns', 1600)

pd.set_option('display.max_rows', 1600)



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler



from sklearn.decomposition import PCA



from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold



from sklearn.metrics import r2_score, mean_squared_error, make_scorer
df = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip", compression='zip')

df_test = pd.read_csv("/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip", compression='zip')
test_id = df_test['ID']

del df_test['ID']
plt.scatter(df['y'], df['ID'])
df = df[df['y']<150]
# Colum names, Missing values count, Missing values percentage in DataFrame

missing_val_df = pd.DataFrame({

    'name': df.columns,

    'mcount': df.isna().sum(),

    'mpercentage': df.isnull().sum()/df.shape[0]*100

})



#missing_val_df.to_csv("missing_values.csv", index=False)

#missing_val_df.sort_values(by='mpercentage', ascending=False)
y = df.iloc[:,1]
y = np.log(y)
X = df.iloc[:, 2:]
X.shape
uniqeCol = []

for col in X.columns:

    if X[col].nunique() == 1:

        uniqeCol.append(col)

        X.drop(columns=col, inplace=True)

        df_test.drop(columns=col, inplace=True)

print(len(uniqeCol))
X.shape
X.info()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.info()
class LabelEncoderExt(object):

    def __init__(self):

        """

        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]

        Unknown will be added in fit and transform will take care of new item. It gives unknown class id

        """

        self.label_encoder = LabelEncoder()

        # self.classes_ = self.label_encoder.classes_



    def fit(self, data_list):

        """

        This will fit the encoder for all the unique values and introduce unknown value

        :param data_list: A list of string

        :return: self

        """

        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])

        self.classes_ = self.label_encoder.classes_



        return self



    def transform(self, data_list):

        """

        This will transform the data_list to id list where the new values get assigned to Unknown class

        :param data_list:

        :return:

        """

        new_data_list = list(data_list)

        for unique_item in np.unique(data_list):

            if unique_item not in self.label_encoder.classes_:

                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]



        return self.label_encoder.transform(new_data_list)
Le = LabelEncoderExt()
for col in X_train.columns:

    if X_train[col].dtype == 'object':

        Le.fit(X_train[col])

        X_train[col] = Le.transform(X_train[col])

        X_test[col] = Le.transform(X_test[col])

        df_test[col] = Le.transform(df_test[col])
# perform PCA

pca = PCA()

X_train_pca_df = pd.DataFrame(pca.fit_transform(X_train))
#pca.explained_variance_

#pca.explained_variance_ratio_
plt.figure(figsize=(12,12))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel("Number of Principal Components")

plt.ylabel("Explained veriance ratio")
pca1 = PCA(n_components=45)
X_train.shape
df_test.shape
X_train_pca_df = pd.DataFrame(pca1.fit_transform(X_train), columns=list(range(0,45)))

X_test_pca_df =  pd.DataFrame(pca1.transform(X_test))

X_test_pca_dfNew =  pd.DataFrame(pca1.transform(df_test))
columns = []

print ("cName","\t", "PearSon Coree","\t\t", "pvalue","\n")

for col in X_train_pca_df.columns:

    corre, pvalue = stats.pearsonr(X_train_pca_df[col], y_train)

    

    if (pvalue < 0.03):

        print (col,"\t", corre*100,"\t", pvalue)

        columns.append(col)

print("\n No of PCAs Highly Correlate with Target Veriable: ==>:", len(columns), "Outof_Available PCAs",X_train_pca_df.shape[1])
X_train_pca_df_final = X_train_pca_df[columns]

X_test_pca_df_final = X_test_pca_df[columns]
X_test_pca_dfNewKaggle = X_test_pca_dfNew[columns]
mMS = MinMaxScaler()
for col in X_test_pca_df_final.columns:

    X_train_pca_df_final[col] = mMS.fit_transform(np.array(X_train_pca_df_final[col]).reshape(-1,1))

    X_test_pca_df_final[col] = mMS.transform(np.array(X_test_pca_df_final[col]).reshape(-1,1))

    X_test_pca_dfNewKaggle[col] = mMS.transform(np.array(X_test_pca_dfNewKaggle[col]).reshape(-1,1))
def model_Execute(xtrain, xtest, ytrain, ytest, model):

    obj = model

    obj.fit(xtrain, ytrain)

    y_predict = obj.predict(xtest)

    y_predict_train = obj.predict(xtrain)

    

    test_r2_score = r2_score(ytest, y_predict)

    train_r2_score = r2_score(ytrain, y_predict_train)

    print(str(obj).split("(")[0])

    print("Train Accuracy(R2 Score): ===========>", train_r2_score)

    print("Test Accuracuy(R2 Score): ===========>", test_r2_score)

    print("MeanSquareError: ====================>", mean_squared_error(ytest, y_predict))

    print("RootMeanSquareError: ================>", np.sqrt(mean_squared_error(ytest, y_predict)), "\n")
models = {

    'lr': LinearRegression(),

    'dt': DecisionTreeRegressor(),

    'rf': RandomForestRegressor(),

    'agb': AdaBoostRegressor(),

    'gbr': GradientBoostingRegressor()

}
for mName, Model in models.items():

    model_Execute(X_train_pca_df_final, X_test_pca_df_final, y_train, y_test, Model)
linRig = LinearRegression()

linRig.fit(X_train_pca_df_final, y_train)
result_df = pd.DataFrame(columns=['ID', 'y'])

result_df['ID'] = test_id

result_df['y'] = np.exp(linRig.predict(X_test_pca_dfNewKaggle))
result_df.to_csv("Submission_File.csv", index=False, sep = ',', encoding = 'utf8')