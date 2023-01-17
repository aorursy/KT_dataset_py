import numpy

import pandas

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier





import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error







from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.constraints import maxnorm



# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)

# load dataset

dataframe = pandas.read_csv("../input/adult.csv")



dataframe = dataframe.replace({'?': numpy.nan}).dropna()



# Assign names to Columns

dataframe.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']



# Encode Data

dataframe.workclass.replace(('Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'),(1,2,3,4,5,6,7,8), inplace=True)

dataframe.education.replace(('Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16), inplace=True)

dataframe.marital_status.replace(('Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'),(1,2,3,4,5,6,7), inplace=True)

dataframe.occupation.replace(('Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14), inplace=True)

dataframe.relationship.replace(('Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'),(1,2,3,4,5,6), inplace=True)

dataframe.race.replace(('White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'),(1,2,3,4,5), inplace=True)

dataframe.sex.replace(('Female', 'Male'),(1,2), inplace=True)

dataframe.native_country.replace(('United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'),(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41), inplace=True)

dataframe.income.replace(('<=50K', '>50K'),(0,1), inplace=True)
print("Head:", dataframe.head())
print("Statistical Description:", dataframe.describe())
print("Shape:", dataframe.shape)
print("Data Types:", dataframe.dtypes)
print("Correlation:", dataframe.corr(method='pearson'))
dataset = dataframe.values





X = dataset[:,0:14]

Y = dataset[:,14] 
# feature extraction

test = SelectKBest(score_func=chi2, k=3)

fit = test.fit(X, Y)



# scores

numpy.set_printoptions(precision=3)

print(fit.scores_)

features = fit.transform(X)



# summarise selected features

print(features[0:10,:])
#Feature Selection

model = LogisticRegression()

rfe = RFE(model, 3)

fit = rfe.fit(X, Y)



print("Number of Features: ", fit.n_features_)

print("Selected Features: ", fit.support_)

print("Feature Ranking: ", fit.ranking_)
pca = PCA(n_components=3)

fit = pca.fit(X)



print("Explained Varience: ", fit.explained_variance_ratio_)
model = ExtraTreesClassifier()

model.fit(X, Y)

print("Feature Importance: ", model.feature_importances_)

plt.hist((dataframe.income))
dataframe.hist()
dataframe.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False)
scatter_matrix(dataframe)
fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,15,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(dataframe.columns)

ax.set_yticklabels(dataframe.columns)
# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)

print("X_Train: ", X_Train.shape)

print("X_Test: ", X_Test.shape)

print("Y_Train: ", Y_Train.shape)

print("Y_Test: ", Y_Test.shape)




num_instances = len(X)



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('L_SVM', LinearSVC()))

models.append(('SGDC', SGDClassifier()))



# Evaluations

results = []

names = []



for name, model in models:

    # Fit the model

    model.fit(X_Train, Y_Train)

    

    predictions = model.predict(X_Test)

    

    # Evaluate the model

    score = accuracy_score(Y_Test, predictions)

    mse = mean_squared_error(predictions, Y_Test)

    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    results.append(mse)

    names.append(name)

    

    msg = "%s: %f (%f)" % (name, score, mse)

    print(msg)

    
# create model

model = Sequential()

model.add(Dense(28, input_dim=14, activation='relu', kernel_initializer="uniform"))

model.add(Dropout(0.2))

model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3), kernel_initializer="uniform"))

model.add(Dropout(0.2))

model.add(Dense(10, activation='relu', kernel_initializer="uniform"))

model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(X_Train, Y_Train, epochs=300, batch_size=10)



# Evaluate the model

scores = model.evaluate(X_Test, Y_Test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))