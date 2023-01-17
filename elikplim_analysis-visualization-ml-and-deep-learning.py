import numpy

import pandas

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier





import matplotlib.pyplot as plt

from pandas.tools.plotting import scatter_matrix



from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

#from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_squared_error





from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from sklearn.preprocessing import LabelEncoder

from keras.constraints import maxnorm





# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)





# load dataset

dataframe = pandas.read_csv("../input/bank-additional.csv")



#dataframe = dataframe.replace({'?': numpy.nan}).dropna()


# Encode Data

dataframe.job.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)



dataframe.marital.replace(('divorced','married','single','unknown'),(1,2,3,4), inplace=True)

dataframe.education.replace(('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'),(1,2,3,4,5,6,7,8), inplace=True)

dataframe.default.replace(('no','yes','unknown'),(1,2,3), inplace=True)

dataframe.housing.replace(('no','yes','unknown'),(1,2,3), inplace=True)

dataframe.loan.replace(('no','yes','unknown'),(1,2,3), inplace=True)

dataframe.contact.replace(('cellular','telephone'),(1,2), inplace=True)

dataframe.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)

dataframe.day_of_week.replace(('mon','tue','wed','thu','fri'),(1,2,3,4,5), inplace=True)

dataframe.poutcome.replace(('failure','nonexistent','success'),(1,2,3), inplace=True)

dataframe.y.replace(('yes','no'),(0,1), inplace=True)



dataframe = dataframe.abs()
print("Head:", dataframe.head())
print("Statistical Description:", dataframe.describe())
print("Shape:", dataframe.shape)
print("Data Types:", dataframe.dtypes)
print("Correlation:", dataframe.corr(method='pearson'))
dataset = dataframe.values





X = dataset[:,0:20]

Y = dataset[:,20] 

# feature extraction

test = SelectKBest(score_func=f_classif, k=3)

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

plt.hist(dataframe.y)
dataframe.hist()
dataframe.plot(kind='density', subplots=True, layout=(4,6), sharex=False, sharey=False)
dataframe.plot(kind='box', subplots=True, layout=(4,6), sharex=False, sharey=False)
scatter_matrix(dataframe)
fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(dataframe.corr(), vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,20,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(dataframe.columns)

ax.set_yticklabels(dataframe.columns)



#plt.show()
# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2)



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

models.append(('ETC', ExtraTreesClassifier()))

models.append(('RFC', RandomForestClassifier()))



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

    

    

#Encode Categorical data

columns_encode = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']

encoded_dataset = pandas.get_dummies(dataframe,columns=columns_encode)

encoded_dataset = encoded_dataset.values





X = encoded_dataset[:,0:63].astype(float)





Y_dataframe = dataframe.values

Y = Y_dataframe[:,20]



encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

print(encoded_Y.shape)



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, encoded_Y, test_size=0.3)

print("X_Train: ", X_Train.shape)

print("X_Test: ", X_Test.shape)

print("Y_Train: ", Y_Train.shape)

print("Y_Test: ", Y_Test.shape)
# create model

model = Sequential()

model.add(Dense(40, input_dim=63, init='uniform', activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(20, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))

model.add(Dropout(0.2))

model.add(Dense(10, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='relu'))



# Compile model

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(X_Train, Y_Train, epochs=100, batch_size=10)



# Evaluate the model

scores = model.evaluate(X_Test, Y_Test)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))