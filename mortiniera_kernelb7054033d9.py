import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="ticks", color_codes=True)

pd.set_option('display.max_columns', None)
#customers_df = pd.read_csv('Telco-Customer-Churn.csv')

customers_df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

customers_df = customers_df[customers_df['TotalCharges'] != ' ']

customers_df.TotalCharges = customers_df.TotalCharges.astype('float')

nb_customers = len(customers_df.index)

print('There are a total of %s customers in the dataset among which %s left within the last month.' %(nb_customers, customers_df[customers_df['Churn'] == "Yes"].shape[0]))

churnNB = customers_df['Churn'].value_counts()[1]

churnrate = float(churnNB) / nb_customers

print('The churn rate is {:.2f}%'.format(churnrate*100))
customers_df.head()
customers_df.describe()
customers_df.SeniorCitizen = customers_df.SeniorCitizen.astype('category')
loyal_customers = customers_df[customers_df['Churn'] == "No"]

disloyal_customers = customers_df[customers_df['Churn'] == "Yes"]
dims = (20, 10)

fig, ax =plt.subplots(2,3,figsize=dims)

plt.suptitle('Histograms of charges and tenure between loyal (first row) and disloyal customers (second row) ')

#loyal customers

sns.distplot(loyal_customers.MonthlyCharges, ax=ax[0, 0])

sns.distplot(loyal_customers.TotalCharges, ax=ax[0, 1])

sns.distplot(loyal_customers.tenure, ax=ax[0,2])

#disloyal customers

sns.distplot(disloyal_customers.MonthlyCharges, ax=ax[1, 0], color='red')

sns.distplot(disloyal_customers.TotalCharges, ax=ax[1, 1], color='red')

sns.distplot(disloyal_customers.tenure, ax=ax[1,2], color='red')

plt.savefig('charges_and_tenure.jpg')
dims = (20, 10)

fig, ax =plt.subplots(2,3,figsize=dims)

plt.suptitle('Pie charts of contracts and phone services between loyal (first row) and disloyal customers (second row) ')

#loyal customers

loyal_customers.Contract.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[0,0], colors = ['blue', 'purple', 'green'])

loyal_customers.InternetService.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[0,1], colors = ['blue', 'purple', 'green'])

loyal_customers.PhoneService.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[0,2], colors = ['blue', 'purple', 'green'])

#disloyal customers

disloyal_customers.Contract.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[1,0], colors = ['red', 'pink', 'orange'])

disloyal_customers.InternetService.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[1,1], colors = ['red', 'pink', 'orange'])

disloyal_customers.PhoneService.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[1,2], colors = ['red', 'pink', 'orange'])

plt.savefig('services and contracts.jpg')
dims = (20, 10)

fig, ax =plt.subplots(2,4,figsize=dims)

plt.suptitle('Customers segmentation between loyal (first row) and disloyal customers (second row) ')

#loyal customers

sns.catplot(x="PhoneService", y="MonthlyCharges", kind="box", hue="Partner", data=loyal_customers, ax=ax[0,0], palette = ['blue', 'purple', 'green'])

loyal_customers[loyal_customers['Partner'] == 'Yes'].MultipleLines.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[0,1], title='Customers with a partner', colors = ['blue', 'purple', 'green'])

loyal_customers[loyal_customers['MonthlyCharges'] < 40].SeniorCitizen.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[0,2], title='Low Cost customers', colors = ['blue', 'purple', 'green'])

loyal_customers[loyal_customers['MonthlyCharges'] > 70].SeniorCitizen.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[0,3], title='Premium customers', colors = ['blue', 'purple', 'green'])

#disloyal customers

sns.catplot(x="PhoneService", y="MonthlyCharges", kind="box", hue="Partner", data=disloyal_customers, ax=ax[1,0], palette = ['red', 'pink', 'orange'])

disloyal_customers[disloyal_customers['Partner'] == 'Yes'].MultipleLines.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[1,1], title='Customers with a partner', colors = ['red', 'pink', 'orange'])

disloyal_customers[disloyal_customers['MonthlyCharges'] < 40].SeniorCitizen.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[1,2], title='Low Cost customers', colors = ['red', 'pink', 'orange'])

disloyal_customers[disloyal_customers['MonthlyCharges'] > 70].SeniorCitizen.value_counts().plot(kind='pie',shadow=True,autopct='%1.1f%%', ax=ax[1,3], title='Premium customers', colors = ['red', 'pink', 'orange'])

plt.close(2)

plt.close(3)

plt.close(4)

plt.close(5)

plt.savefig('multi_variate.jpg')
print("Proportion of senior citizen in whole database %s" %(customers_df.SeniorCitizen.value_counts().values / customers_df.shape[0]))

print("Proportion of senior citizen among loyal customers %s" %(loyal_customers.SeniorCitizen.value_counts().values / loyal_customers.shape[0]))

print("Proportion of senior citizen among disloyal customers %s" %(disloyal_customers.SeniorCitizen.value_counts().values / disloyal_customers.shape[0]))
data = customers_df.copy()



#Convert gender to binary and drop redundant column

data["Male"]=data['gender'].map(lambda x : 1  if x =='Male' else 0)

data = data.drop(columns="gender")



#Convert internetService to binary : 1 if there is any, else 0. Then we create a column for fiber optic,

# the negative option will automatically imply DSL. Hence we can remove internet service column

data["InternetYes"]= data['InternetService'].map(lambda x :0  if x =='No' else 1)

data["FiberOptic"]= data["InternetService"].map(lambda x : 1  if x =='Fiber optic' else 0)

data = data.drop(columns="InternetService")



#Convert target variable to binary

data["Churn"]= data['Churn'].map(lambda x : 0  if x =='No' else 1)



binary_columns=["Partner","Dependents","PhoneService","MultipleLines","PaperlessBilling","OnlineSecurity",

                "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]



for c in binary_columns:

    data[c] = data[c].map(lambda x : 1  if x =='Yes' else 0)

    

#Create dummies for the remaining categorical columns and drop redundant original column

data = pd.concat([data, pd.get_dummies(data["Contract"],prefix="Contract")], axis=1)

data = data.drop(columns="Contract")



data = pd.concat([data, pd.get_dummies(data["PaymentMethod"],prefix="Pay")], axis=1)

data= data.drop(columns="PaymentMethod")



#finally drop customerID columns as it is irrelevant information for the model

data = data.drop(columns="customerID")
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

data = pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve, f1_score, recall_score, accuracy_score, precision_score, confusion_matrix
def print_confusion_matrix(y_test, y_pred) :   

    

    conf_matrix = pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[1,0]), columns = ['Churn Yes', 'Churn No'])

    conf_matrix.index = ['Churn Yes', 'Churn No']

    

    print("Accuracy Score:",accuracy_score(y_test,y_pred))

    print("Recall Score:",recall_score(y_test,y_pred,labels=[1,0]))

    print("Precision Score:",precision_score(y_test,y_pred,labels=[1,0]))

    print("Confusion Matrix:")

    

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")

    plt.xlabel('Predicted Labels')

    plt.ylabel('True labels')
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns="Churn"), data["Churn"], stratify=data["Churn"], random_state=42)

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, stratify=y_test, random_state=42)

everyone_churn = np.ones_like(y_test)

print_confusion_matrix(y_test, everyone_churn)

plt.savefig('everyone_churn.png')
nobody_churn = np.zeros_like(y_test)

print_confusion_matrix(y_test, nobody_churn)

plt.savefig('nobody_churn.png')
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

import matplotlib.ticker
param_grid = {'penalty' : ['l1', 'l2'],

    'C' : np.logspace(-1, 1, 10),

    'solver' : ['liblinear']}



clf = LogisticRegression()

gs = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, verbose=0, scoring="recall")

gs.fit(X_train, y_train)



y_score = gs.decision_function(X_valid)



precision, recall, thresholds = precision_recall_curve(y_valid, y_score)
plt.plot(thresholds, precision[:len(precision)-1], label='precision', ls = 'dashed')

plt.plot(thresholds, recall[:len(recall)-1], label='recall', ls = 'dashed')

plt.legend()

plt.title('Precision and Recall scores as a function of the decision threshold')

plt.xlabel('Threshold')

plt.ylabel('Metrics value')

plt.grid()



#axes

ax=plt.gca()



f = lambda x,pos: str(x).rstrip('0').rstrip('.')

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))

plt.savefig('precision_recall_threshold.png')

plt.show()
y_pred_threshold = (gs.decision_function(X_test) >= -1.5).astype(bool) #this computes a new set of y_pred based on a different threshold, which we set on the decision function 



print_confusion_matrix(y_test,y_pred_threshold)

plt.xlabel('Predicted Labels')

plt.ylabel('True labels')

plt.show()
estimator = gs.best_estimator_

class_labels = gs.classes_

weights = estimator.coef_[0]

weights_index = np.argsort(weights)[::-1]





#take 5 most important feature in each class

weights = np.sort(weights)[::-1]



#about to churn

positive_class = weights_index[:5]

positive_feature = X_train.columns[positive_class].values

coeff_pos = weights[:5]







negative_class = weights_index[-5:][::-1]

negative_feature = X_train.columns[negative_class].values

coeff_neg = weights[-5:][::-1]





top5_class1 = list(zip(coeff_pos, positive_feature))

top5_class2 = list(zip(coeff_neg, negative_feature))



print("Most important feature used to predict churn with their weights")

print('--------------------------------------')



for w, n in top5_class1 :

    print("{} : {}".format(n, w))

    

print('-----------')

    

for w, n in top5_class2 :

    print("{} : {}".format(n, w))

print("Number of customers : %s" % y_test.shape[0])

print("Number of churning customers : %s" % sum(y_test == 1))
def cost(clients_to_retain, clients_to_obtain) :

    return 50*clients_to_retain + 200*clients_to_obtain
nobody_churn = cost(0, 117)

everyone_churn = cost(440, 0)

logit_model = cost(106, 11)



strategies = [(nobody_churn, "No action taken"), (everyone_churn, "Retain all customers"), (logit_model, "Our model")]



all_costs = pd.DataFrame(strategies, columns=["Cost ($)", "Strategy"])

all_costs.head()
sns.catplot(x="Strategy", y="Cost ($)", kind="bar", data=all_costs, palette=["red", "yellow", "green"])

plt.title("Associated cost for each strategy taken")

plt.savefig("strategies_cost.png")

plt.grid()



ax=plt.gca()

f = lambda x,pos: str(x).rstrip('0').rstrip('.')

ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2000))

ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))



plt.show()