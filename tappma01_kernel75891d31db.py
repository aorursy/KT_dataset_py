# Load libraries

import pandas as pd

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import preprocessing

from sklearn import utils







df = pd.read_csv("../input/GlobalLandTemperaturesByCountry.csv")

df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

dataset = df[df.dt >= "1800"]

datasetCleaned = dataset.drop("AverageTemperatureUncertainty", axis=1)

datasetCleaned

datasetCleaned['dt'] = datasetCleaned['dt'].apply(lambda x:x[:4])

# datasetCleaned.groupby(['dt']).mean()



df = datasetCleaned.groupby(['dt','Country'])

ndf = df['AverageTemperature'].mean().reset_index()

ndf.dt = ndf.dt.astype('int64')



ndf = pd.get_dummies(ndf)



ndf = ndf[['dt', 'Country_Afghanistan', 'Country_Africa', 'Country_Albania', 'Country_Algeria', 'Country_American Samoa', 'Country_Andorra', 'Country_Angola', 'Country_Anguilla', 'Country_Antigua And Barbuda', 'Country_Argentina', 'Country_Armenia', 'Country_Aruba', 'Country_Asia', 'Country_Australia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Bahamas', 'Country_Bahrain', 'Country_Baker Island', 'Country_Bangladesh', 'Country_Barbados', 'Country_Belarus', 'Country_Belgium', 'Country_Belize', 'Country_Benin', 'Country_Bhutan', 'Country_Bolivia', 'Country_Bonaire, Saint Eustatius And Saba', 'Country_Bosnia And Herzegovina', 'Country_Botswana', 'Country_Brazil', 'Country_British Virgin Islands', 'Country_Bulgaria', 'Country_Burkina Faso', 'Country_Burma', 'Country_Burundi', 'Country_Cambodia', 'Country_Cameroon', 'Country_Canada', 'Country_Cape Verde', 'Country_Cayman Islands', 'Country_Central African Republic', 'Country_Chad', 'Country_Chile', 'Country_China', 'Country_Christmas Island', 'Country_Colombia', 'Country_Comoros', 'Country_Congo', 'Country_Congo (Democratic Republic Of The)', 'Country_Costa Rica', 'Country_Croatia', 'Country_Cuba', 'Country_Curaçao', 'Country_Cyprus', 'Country_Czech Republic', "Country_Côte D'Ivoire", 'Country_Denmark', 'Country_Denmark (Europe)', 'Country_Djibouti', 'Country_Dominica', 'Country_Dominican Republic', 'Country_Ecuador', 'Country_Egypt', 'Country_El Salvador', 'Country_Equatorial Guinea', 'Country_Eritrea', 'Country_Estonia', 'Country_Ethiopia', 'Country_Europe', 'Country_Falkland Islands (Islas Malvinas)', 'Country_Faroe Islands', 'Country_Federated States Of Micronesia', 'Country_Fiji', 'Country_Finland', 'Country_France', 'Country_France (Europe)', 'Country_French Guiana', 'Country_French Polynesia', 'Country_French Southern And Antarctic Lands', 'Country_Gabon', 'Country_Gambia', 'Country_Gaza Strip', 'Country_Georgia', 'Country_Germany', 'Country_Ghana', 'Country_Greece', 'Country_Greenland', 'Country_Grenada', 'Country_Guadeloupe', 'Country_Guam', 'Country_Guatemala', 'Country_Guernsey', 'Country_Guinea', 'Country_Guinea Bissau', 'Country_Guyana', 'Country_Haiti', 'Country_Heard Island And Mcdonald Islands', 'Country_Honduras', 'Country_Hong Kong', 'Country_Hungary', 'Country_Iceland', 'Country_India', 'Country_Indonesia', 'Country_Iran', 'Country_Iraq', 'Country_Ireland', 'Country_Isle Of Man', 'Country_Israel', 'Country_Italy', 'Country_Jamaica', 'Country_Japan', 'Country_Jersey', 'Country_Jordan', 'Country_Kazakhstan', 'Country_Kenya', 'Country_Kingman Reef', 'Country_Kiribati', 'Country_Kuwait', 'Country_Kyrgyzstan', 'Country_Laos', 'Country_Latvia', 'Country_Lebanon', 'Country_Lesotho', 'Country_Liberia', 'Country_Libya', 'Country_Liechtenstein', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Macau', 'Country_Macedonia', 'Country_Madagascar', 'Country_Malawi', 'Country_Malaysia', 'Country_Mali', 'Country_Malta', 'Country_Martinique', 'Country_Mauritania', 'Country_Mauritius', 'Country_Mayotte', 'Country_Mexico', 'Country_Moldova', 'Country_Monaco', 'Country_Mongolia', 'Country_Montenegro', 'Country_Montserrat', 'Country_Morocco', 'Country_Mozambique', 'Country_Namibia', 'Country_Nepal', 'Country_Netherlands', 'Country_Netherlands (Europe)', 'Country_New Caledonia', 'Country_New Zealand', 'Country_Nicaragua', 'Country_Niger', 'Country_Nigeria', 'Country_Niue', 'Country_North America', 'Country_North Korea', 'Country_Northern Mariana Islands', 'Country_Norway', 'Country_Oceania', 'Country_Oman', 'Country_Pakistan', 'Country_Palau', 'Country_Palestina', 'Country_Palmyra Atoll', 'Country_Panama', 'Country_Papua New Guinea', 'Country_Paraguay', 'Country_Peru', 'Country_Philippines', 'Country_Poland', 'Country_Portugal', 'Country_Puerto Rico', 'Country_Qatar', 'Country_Reunion', 'Country_Romania', 'Country_Russia', 'Country_Rwanda', 'Country_Saint Barthélemy', 'Country_Saint Kitts And Nevis', 'Country_Saint Lucia', 'Country_Saint Martin', 'Country_Saint Pierre And Miquelon', 'Country_Saint Vincent And The Grenadines', 'Country_Samoa', 'Country_San Marino', 'Country_Sao Tome And Principe', 'Country_Saudi Arabia', 'Country_Senegal', 'Country_Serbia', 'Country_Seychelles', 'Country_Sierra Leone', 'Country_Singapore', 'Country_Sint Maarten', 'Country_Slovakia', 'Country_Slovenia', 'Country_Solomon Islands', 'Country_Somalia', 'Country_South Africa', 'Country_South America', 'Country_South Georgia And The South Sandwich Isla', 'Country_South Korea', 'Country_Spain', 'Country_Sri Lanka', 'Country_Sudan', 'Country_Suriname', 'Country_Svalbard And Jan Mayen', 'Country_Swaziland', 'Country_Sweden', 'Country_Switzerland', 'Country_Syria', 'Country_Taiwan', 'Country_Tajikistan', 'Country_Tanzania', 'Country_Thailand', 'Country_Timor Leste', 'Country_Togo', 'Country_Tonga', 'Country_Trinidad And Tobago', 'Country_Tunisia', 'Country_Turkey', 'Country_Turkmenistan', 'Country_Turks And Caicas Islands', 'Country_Uganda', 'Country_Ukraine', 'Country_United Arab Emirates', 'Country_United Kingdom', 'Country_United Kingdom (Europe)', 'Country_United States', 'Country_Uruguay', 'Country_Uzbekistan', 'Country_Venezuela', 'Country_Vietnam', 'Country_Virgin Islands', 'Country_Western Sahara', 'Country_Yemen', 'Country_Zambia', 'Country_Zimbabwe', 'Country_Åland', 'AverageTemperature']]

ndf['AverageTemperature'] = ndf['AverageTemperature'].astype(int)



# Split-out validation dataset

array = ndf.values

X = array[:,0:-1]

Y = array[:,-1]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)





# Test options and evaluation metric

seed = 7

scoring = 'accuracy'



# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Make predictions on validation dataset

LDA = LinearDiscriminantAnalysis()

LDA.fit(X_train, Y_train)

predictions = LDA.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
# Make predictions on validation dataset

NB = GaussianNB()

NB.fit(X_train, Y_train)

predictions = NB.predict(X_validation)

print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
# Make predictions on validation dataset

FM = DecisionTreeClassifier()

FM.fit(X, Y)

Xnew = [[2014,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],

        [2014,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

        [2014,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],

        [1800,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

ynew = FM.predict(Xnew)

print(ynew)