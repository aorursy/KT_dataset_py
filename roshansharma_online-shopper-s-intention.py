# for basic operations

import numpy as np

import pandas as pd



# for getting the file path

import os

print(os.listdir('../input'))



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go
# reading the dataset



data = pd.read_csv('../input/online_shoppers_intention.csv')



# checking the shape of the data

data.shape
# checking the head of the data



data.head()
# describing the data



data.describe()
# checking the percentage of missing data contains in all the columns



missing_percentage = data.isnull().sum()/data.shape[0]

print(missing_percentage)
# checking the Distribution of customers on Revenue



plt.rcParams['figure.figsize'] = (18, 7)



plt.subplot(1, 2, 1)

sns.countplot(data['Weekend'], palette = 'pastel')

plt.title('Buy or Not', fontsize = 30)

plt.xlabel('Revenue or not', fontsize = 15)

plt.ylabel('count', fontsize = 15)





# checking the Distribution of customers on Weekend

plt.subplot(1, 2, 2)

sns.countplot(data['Weekend'], palette = 'inferno')

plt.title('Purchase on Weekends', fontsize = 30)

plt.xlabel('Weekend or not', fontsize = 15)

plt.ylabel('count', fontsize = 15)



plt.show()
data['VisitorType'].value_counts()
# plotting a pie chart for browsers



plt.rcParams['figure.figsize'] = (18, 7)

size = [10551, 1694, 85]

colors = ['violet', 'magenta', 'pink']

labels = "Returning Visitor", "New_Visitor", "Others"

explode = [0, 0, 0.1]

plt.subplot(1, 2, 1)

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')

plt.title('Different Visitors', fontsize = 30)

plt.axis('off')

plt.legend()



# plotting a pie chart for browsers

size = [7961, 2462, 736, 467,174, 163, 300]

colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'blue']

labels = "2", "1","4","5","6","10","others"



plt.subplot(1, 2, 2)

plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle = 90)

plt.title('Different Browsers', fontsize = 30)

plt.axis('off')

plt.legend()

plt.show()
# visualizing the distribution of customers around the Region



plt.rcParams['figure.figsize'] = (18, 7)



plt.subplot(1, 2, 1)

plt.hist(data['TrafficType'], color = 'lightgreen')

plt.title('Distribution of diff Traffic',fontsize = 30)

plt.xlabel('TrafficType Codes', fontsize = 15)

plt.ylabel('Count', fontsize = 15)



# visualizing the distribution of customers around the Region



plt.subplot(1, 2, 2)

plt.hist(data['Region'], color = 'lightblue')

plt.title('Distribution of Customers',fontsize = 30)

plt.xlabel('Region Codes', fontsize = 15)

plt.ylabel('Count', fontsize = 15)



plt.show()
# checking the no. of OSes each user is having



data['OperatingSystems'].value_counts()
#checking the months with most no.of customers visiting the online shopping sites



data['Month'].value_counts()
# creating a donut chart for the months variations'



# plotting a pie chart for different number of OSes users have.



size = [6601, 2585, 2555, 478, 111]

colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen']

labels = "2", "1","3","4","others"

explode = [0, 0, 0, 0, 0]



circle = plt.Circle((0, 0), 0.6, color = 'white')



plt.subplot(1, 2, 1)

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')

plt.title('OSes Users have', fontsize = 30)

p = plt.gcf()

p.gca().add_artist(circle)

plt.axis('off')

plt.legend()



# plotting a pie chart for share of special days



size = [3364, 2998, 1907, 1727, 549, 448, 433, 432, 288, 184]

colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'magenta', 'lightblue', 'lightgreen', 'violet']

labels = "May", "November", "March", "December", "October", "September", "August", "July", "June", "February"

explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



circle = plt.Circle((0, 0), 0.6, color = 'white')



plt.subplot(1, 2, 2)

plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')

plt.title('Special Days', fontsize = 30)

p = plt.gcf()

p.gca().add_artist(circle)

plt.axis('off')

plt.legend()



plt.show()
# product related duration vs revenue



plt.rcParams['figure.figsize'] = (18, 15)



plt.subplot(2, 2, 1)

sns.boxenplot(data['Revenue'], data['Informational_Duration'], palette = 'rainbow')

plt.title('Info. duration vs Revenue', fontsize = 30)

plt.xlabel('Info. duration', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)



# product related duration vs revenue



plt.subplot(2, 2, 2)

sns.boxenplot(data['Revenue'], data['Administrative_Duration'], palette = 'pastel')

plt.title('Admn. duration vs Revenue', fontsize = 30)

plt.xlabel('Admn. duration', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)



# product related duration vs revenue



plt.subplot(2, 2, 3)

sns.boxenplot(data['Revenue'], data['ProductRelated_Duration'], palette = 'dark')

plt.title('Product Related duration vs Revenue', fontsize = 30)

plt.xlabel('Product Related duration', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)



# exit rate vs revenue



plt.subplot(2, 2, 4)

sns.boxenplot(data['Revenue'], data['ExitRates'], palette = 'spring')

plt.title('ExitRates vs Revenue', fontsize = 30)

plt.xlabel('ExitRates', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)





plt.show()

# page values vs revenue



plt.rcParams['figure.figsize'] = (18, 7)



plt.subplot(1, 2, 1)

sns.stripplot(data['Revenue'], data['PageValues'], palette = 'autumn')

plt.title('PageValues vs Revenue', fontsize = 30)

plt.xlabel('PageValues', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)



# bounce rates vs revenue

plt.subplot(1, 2, 2)

sns.stripplot(data['Revenue'], data['BounceRates'], palette = 'magma')

plt.title('Bounce Rates vs Revenue', fontsize = 30)

plt.xlabel('Boune Rates', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)



plt.show()
# weekend vs Revenue



df = pd.crosstab(data['Weekend'], data['Revenue'])

df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])

plt.title('Weekend vs Revenue', fontsize = 30)

plt.show()
# Traffic Type vs Revenue



df = pd.crosstab(data['TrafficType'], data['Revenue'])

df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightpink', 'yellow'])

plt.title('Traffic Type vs Revenue', fontsize = 30)

plt.show()
# visitor type vs revenue



df = pd.crosstab(data['VisitorType'], data['Revenue'])

df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightgreen', 'green'])

plt.title('Visitor Type vs Revenue', fontsize = 30)

plt.show()

# region vs Revenue



df = pd.crosstab(data['Region'], data['Revenue'])

df.div(df.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['lightblue', 'blue'])

plt.title('Region vs Revenue', fontsize = 30)

plt.show()
# lm plot



plt.rcParams['figure.figsize'] = (20, 10)



sns.lmplot(x = 'Administrative', y = 'Informational', data = data, x_jitter = 0.05)

plt.title('LM Plot between Admistrative and Information', fontsize = 15)

# month vs pagevalues wrt revenue



plt.rcParams['figure.figsize'] = (18, 15)

plt.subplot(2, 2, 1)

sns.boxplot(x = data['Month'], y = data['PageValues'], hue = data['Revenue'], palette = 'inferno')

plt.title('Mon. vs PageValues w.r.t. Rev.', fontsize = 30)



# month vs exitrates wrt revenue

plt.subplot(2, 2, 2)

sns.boxplot(x = data['Month'], y = data['ExitRates'], hue = data['Revenue'], palette = 'Reds')

plt.title('Mon. vs ExitRates w.r.t. Rev.', fontsize = 30)



# month vs bouncerates wrt revenue

plt.subplot(2, 2, 3)

sns.boxplot(x = data['Month'], y = data['BounceRates'], hue = data['Revenue'], palette = 'Oranges')

plt.title('Mon. vs BounceRates w.r.t. Rev.', fontsize = 30)



# visitor type vs exit rates w.r.t revenue

plt.subplot(2, 2, 4)

sns.boxplot(x = data['VisitorType'], y = data['BounceRates'], hue = data['Revenue'], palette = 'Purples')

plt.title('Visitors vs BounceRates w.r.t. Rev.', fontsize = 30)



plt.show()
# visitor type vs exit rates w.r.t revenue



plt.rcParams['figure.figsize'] = (18, 15)

plt.subplot(2, 2, 1)

sns.violinplot(x = data['VisitorType'], y = data['ExitRates'], hue = data['Revenue'], palette = 'rainbow')

plt.title('Visitors vs ExitRates wrt Rev.', fontsize = 30)



# visitor type vs exit rates w.r.t revenue

plt.subplot(2, 2, 2)

sns.violinplot(x = data['VisitorType'], y = data['PageValues'], hue = data['Revenue'], palette = 'gnuplot')

plt.title('Visitors vs PageValues wrt Rev.', fontsize = 30)



# region vs pagevalues w.r.t. revenue

plt.subplot(2, 2, 3)

sns.violinplot(x = data['Region'], y = data['PageValues'], hue = data['Revenue'], palette = 'Greens')

plt.title('Region vs PageValues wrt Rev.', fontsize = 30)



#region vs exit rates w.r.t. revenue

plt.subplot(2, 2, 4)

sns.violinplot(x = data['Region'], y = data['ExitRates'], hue = data['Revenue'], palette = 'spring')

plt.title('Region vs Exit Rates w.r.t. Revenue', fontsize = 30)



plt.show()
# Inputing Missing Values with 0



data.fillna(0, inplace = True)



# checking the no. of null values in data after imputing the missing values

data.isnull().sum().sum()
# Q1: Time Spent by The Users on Website vs Bounce Rates



'''

Bounce Rate :The percentage of visitors to a particular website who navigate away from the site after 

viewing only one page.

'''

# let's cluster Administrative duration and Bounce Ratw to different types of clusters in the dataset.

# preparing the dataset

x = data.iloc[:, [1, 6]].values



# checking the shape of the dataset

x.shape





from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i,

              init = 'k-means++',

              max_iter = 300,

              n_init = 10,

              random_state = 0,

              algorithm = 'elkan',

              tol = 0.001)

    km.fit(x)

    labels = km.labels_

    wcss.append(km.inertia_)

    

plt.rcParams['figure.figsize'] = (15, 7)

plt.plot(range(1, 11), wcss)

plt.grid()

plt.tight_layout()

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'General Customers')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'Target Customers')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Administrative Duration vs Duration', fontsize = 20)

plt.grid()

plt.xlabel('Administrative Duration')

plt.ylabel('Bounce Rates')

plt.legend()

plt.show()
# informational duration vs Bounce Rates

x = data.iloc[:, [3, 6]].values



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i,

              init = 'k-means++',

              max_iter = 300,

              n_init = 10,

              random_state = 0,

              algorithm = 'elkan',

              tol = 0.001)

    km.fit(x)

    labels = km.labels_

    wcss.append(km.inertia_)

    

plt.rcParams['figure.figsize'] = (15, 7)

plt.plot(range(1, 11), wcss)

plt.grid()

plt.tight_layout()

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Informational Duration vs Bounce Rates', fontsize = 20)

plt.grid()

plt.xlabel('Informational Duration')

plt.ylabel('Bounce Rates')

plt.legend()

plt.show()
# informational duration vs Bounce Rates

x = data.iloc[:, [1, 7]].values



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i,

              init = 'k-means++',

              max_iter = 300,

              n_init = 10,

              random_state = 0,

              algorithm = 'elkan',

              tol = 0.001)

    km.fit(x)

    labels = km.labels_

    wcss.append(km.inertia_)

    

plt.rcParams['figure.figsize'] = (15, 7)

plt.plot(range(1, 11), wcss)

plt.grid()

plt.tight_layout()

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Administrative Clustering vs Exit Rates', fontsize = 20)

plt.grid()

plt.xlabel('Administrative Duration')

plt.ylabel('Exit Rates')

plt.legend()

plt.show()
# informational duration vs Bounce Rates

x = data.iloc[:, [13, 14]].values



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i,

              init = 'k-means++',

              max_iter = 300,

              n_init = 10,

              random_state = 0,

              algorithm = 'elkan',

              tol = 0.001)

    km.fit(x)

    labels = km.labels_

    wcss.append(km.inertia_)

    

plt.rcParams['figure.figsize'] = (15, 7)

plt.plot(range(1, 11), wcss)

plt.grid()

plt.tight_layout()

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Un-interested Customers')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Region vs Traffic Type', fontsize = 20)

plt.grid()

plt.xlabel('Region')

plt.ylabel('Traffic')

plt.legend()

plt.show()
# informational duration vs Bounce Rates

x = data.iloc[:, [1, 13]].values



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i,

              init = 'k-means++',

              max_iter = 300,

              n_init = 10,

              random_state = 0,

              algorithm = 'elkan',

              tol = 0.001)

    km.fit(x)

    labels = km.labels_

    wcss.append(km.inertia_)

    

plt.rcParams['figure.figsize'] = (15, 7)

plt.plot(range(1, 11), wcss)

plt.grid()

plt.tight_layout()

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'Unproductive Customers')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'Target Customers')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Adminstrative Duration vs Region', fontsize = 20)

plt.grid()

plt.xlabel('Administrative Duration')

plt.ylabel('Region Type')

plt.legend()

plt.show()
# one hot encoding 



data1 = pd.get_dummies(data)



data1.columns
# label encoding of revenue



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

data['Revenue'] = le.fit_transform(data['Revenue'])

data['Revenue'].value_counts()
# getting dependent and independent variables



x = data1

# removing the target column revenue from x

x = x.drop(['Revenue'], axis = 1)



y = data['Revenue']



# checking the shapes

print("Shape of x:", x.shape)

print("Shape of y:", y.shape)

# splitting the data



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)



# checking the shapes



print("Shape of x_train :", x_train.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_test :", y_test.shape)
# MODELLING



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



model = RandomForestClassifier()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)



# evaluating the model

print("Training Accuracy :", model.score(x_train, y_train))

print("Testing Accuracy :", model.score(x_test, y_test))



# confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (6, 6)

sns.heatmap(cm ,annot = True)



# classification report

cr = classification_report(y_test, y_pred)

print(cr)
# finding the Permutation importance



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state = 0).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
# plotting the partial dependence plot for adminisrative duration



# importing pdp

from pdpbox import pdp, info_plots



base_features = x_test.columns.values.tolist()



feat_name = 'Administrative_Duration'

pdp_dist = pdp.pdp_isolate(model=model, dataset=x_test, model_features = base_features, feature = feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# plotting partial dependency plot for Informational Duration



base_features = x_test.columns.tolist()



feat_name = 'Informational_Duration'

pdp_dist = pdp.pdp_isolate(model, x_test, base_features, feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
# let's take a look at the shap values



# importing shap

import shap



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x_test)



shap.summary_plot(shap_values[1], x_test, plot_type = 'bar')
shap.summary_plot(shap_values[1], x_test)
# let's create a function to check the customer's conditions



def customer_analysis(model, customer):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(customer)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], customer)
# let's do some real time prediction for patients



customers = x_test.iloc[1,:].astype(float)

customer_analysis(model, customers)
# let's do some real time prediction for patients



customers = x_test.iloc[15,:].astype(float)

customer_analysis(model, customers)
# let's do some real time prediction for patients



customers = x_test.iloc[100,:].astype(float)

customer_analysis(model, customers)
# let's do some real time prediction for patients



customers = x_test.iloc[150,:].astype(float)

customer_analysis(model, customers)
# let's do some real time prediction for patients



customers = x_test.iloc[200,:].astype(float)

customer_analysis(model, customers)
shap_values = explainer.shap_values(x_train.iloc[:50])

shap.initjs()



shap.force_plot(explainer.expected_value[1], shap_values[1], x_test.iloc[:50])

























































































































