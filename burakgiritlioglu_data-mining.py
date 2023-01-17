import pandas as pd
import numpy as np
import seaborn as sns  # Provides a high level interface for drawing attractive and informative statistical graphics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
         ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='black', rotation=0, xytext=(0, 10),
         textcoords='offset points')   
            
# warning library
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA
           
%matplotlib inline
sns.set()
from subprocess import check_output


#def annot_plot_num(ax,w,h):                                    # function to add data to plot
 #   ax.spines['top'].set_visible(False)
  #  ax.spines['right'].set_visible(False)
   # for p in ax.patches:
    #    ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))


df = pd.read_csv("/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")
df.Response = df.Response.apply(lambda X : 0 if X == 'No' else 1)
df.head()
df.shape
df.columns
df.describe()
df.dtypes
df.dtypes.groupby(df.dtypes.values).count()
ax = sns.countplot('Response',data = df)
plt.ylabel('Total number of Response')
annot_plot(ax,0.09,1) #I found this function in kaggle examples. I imported the function 'Import Library Part'
plt.show()

def plot_hist(var,tot):
    ax = sns.countplot(var,data = df)
    plt.ylabel(tot)
    annot_plot(ax,0.09,1) #I found this function in kaggle examples. I imported the function 'Import Library Part'
    plt.show()

strvar=["State", "Coverage", "Education", "EmploymentStatus", "Gender","Location Code", "Marital Status", "Policy", "Policy Type", "Renew Offer Type", "Sales Channel", "Vehicle Size", "Vehicle Class"]
total=["Total number of State", "Total number of Coverage", "Total number of Education", "Total number of EmploymentStatus","Total number of Gender", "Total number of Location Code", "Total number of Marital Status", "Total number of Policy",
        "Total number of Policy Type", "Total number of Renew Offer Type","Total number of Sales Channel","Total number of Vehicle Size", "Total number of Vehicle Class"]
for n,i in zip(strvar,total):
    plot_hist(n,i)

def plot_hist(var,tot):
    ax = sns.countplot('Response',hue = var ,data = df)
    plt.ylabel(tot)
    annot_plot(ax,0.09,1)
    plt.show()

strvar=["State", "Coverage", "Education", "EmploymentStatus", "Gender","Location Code", "Marital Status", "Policy", "Policy Type", "Renew Offer Type", "Sales Channel", "Vehicle Size", "Vehicle Class"]
total=["Total number of State", "Total number of Coverage", "Total number of Education", "Total number of EmploymentStatus","Total number of Gender", "Total number of Location Code", "Total number of Marital Status", "Total number of Policy",
        "Total number of Policy Type", "Total number of Renew Offer Type","Total number of Sales Channel","Total number of Vehicle Size", "Total number of Vehicle Class"]
for n,i in zip(strvar,total):
    plot_hist(n,i)
def plot(var):
    plt.figure(figsize=(12,6))
    sns.boxplot(y = var, x = 'Response', data = df)
    plt.ylabel(var)
    plt.show()

numvar=["Income","Total Claim Amount"]
for n in numvar:
    plot(n)

df = df.drop(['Customer','Effective To Date','Gender','Policy','Vehicle Class'], axis = 1)

df["Coverage"] = [0 if i == "Basic" else 1 if i == "Extended"
                      else 2 for i in df["Coverage"]]
df["State"] = [0 if i == "California" else 1 if i == "Oregon" else 2 if i == "Arizona" else 3 if i == "Nevada"
                      else 4 for i in df["State"]]
df["Education"] = [0 if i == "Bachelor" else 1 if i == "College" else 2 if i == "High School or Below" else 3 if i == "College" 
                   else 4 if i == "Master" else 5 for i in df["Education"]]
df["EmploymentStatus"] = [0 if i == "Employed" else 1 if i == "Unemployed" else 2 if i == "Medical Leave" else 3 if i == "Medical Leave" 
                  else 4 if i == "Disabled" else 5 for i in df["EmploymentStatus"]]
df["Location Code"] = [0 if i == "Suburban" else 1 if i == "Rural"
                      else 2 for i in df["Location Code"]]
df["Marital Status"] = [0 if i == "Married" else 1 if i == "Single"
                      else 2 for i in df["Marital Status"]]
df["Policy Type"] = [0 if i == "Personal Auto" else 1 if i == "Corporate Auto"
                      else 2 for i in df["Policy Type"]]
df["Renew Offer Type"] = [0 if i == "Offer1" else 1 if i == "Offer2" else 2 if i == "Offer3"
                      else 3 for i in df["Renew Offer Type"]]
df["Sales Channel"] = [0 if i == "Agent" else 1 if i == "Branch" else 2 if i == "Call Center"
                      else 3 for i in df["Sales Channel"]]   
df["Vehicle Size"] = [0 if i == "Medsize" else 1 if i == "Small" 
                      else 2 for i in df["Vehicle Size"]] 

df.head()


df.info()
def plot(var):
    sns.countplot(x= var,data = df)
    plt.xticks(rotation = 60)
    plt.show()
strvar=["State", "Coverage", "Education", "EmploymentStatus","Location Code", "Marital Status", "Policy Type", "Renew Offer Type", "Sales Channel", "Vehicle Size"]
for i in strvar:
    plot(i)
X1 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type","Renew Offer Type", "Sales Channel",
         "Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model = sm.OLS(y1, X1).fit()
predictions = model.predict(X1)

# Print out the statistics
model.summary()
X1 = df[["Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model1 = sm.OLS(y1, X1).fit()
predictions = model1.predict(X1)

# Print out the statistics
model1.summary()
X1 = df[["Coverage","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model2 = sm.OLS(y1, X1).fit()
predictions = model2.predict(X1)

# Print out the statistics
model2.summary()


X1 = df[["Coverage","Education","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model3 = sm.OLS(y1, X1).fit()
predictions = model3.predict(X1)

# Print out the statistics
model3.summary()
X1 = df[["Coverage","Education","EmploymentStatus","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model4 = sm.OLS(y1, X1).fit()
predictions = model4.predict(X1)

# Print out the statistics
model4.summary()
X1 = df[["Coverage","Education","EmploymentStatus","Marital Status","Renew Offer Type", "Sales Channel","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model5 = sm.OLS(y1, X1).fit()
predictions = model5.predict(X1)

# Print out the statistics
model5.summary()
X1 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type", "Sales Channel","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model6 = sm.OLS(y1, X1).fit()
predictions = model6.predict(X1)

# Print out the statistics
model6.summary()

X1 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type","Vehicle Size"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model7 = sm.OLS(y1, X1).fit()
predictions = model7.predict(X1)

# Print out the statistics
model7.summary()


X1 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel"]]
y1 = df["Response"]

# Fit and make the predictions by the model
model8 = sm.OLS(y1, X1).fit()
predictions = model8.predict(X1)

# Print out the statistics
model8.summary()
X = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
X1 = df[["Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
X2 = df[["Coverage","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
X3 = df[["Coverage","Education","Marital Status","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
X4 = df[["Coverage","Education","EmploymentStatus","Policy Type","Renew Offer Type", "Sales Channel","Vehicle Size"]]
X5 = df[["Coverage","Education","EmploymentStatus","Marital Status","Renew Offer Type", "Sales Channel","Vehicle Size"]]
X6 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type", "Sales Channel","Vehicle Size"]]
X7 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type","Vehicle Size"]]
X8 = df[["Coverage","Education","EmploymentStatus","Marital Status","Policy Type","Renew Offer Type", "Sales Channel"]]
y = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=100)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=100)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.3, random_state=100)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y, test_size=0.3, random_state=100)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y, test_size=0.3, random_state=100)
X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y, test_size=0.3, random_state=100)
X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y, test_size=0.3, random_state=100)
X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y, test_size=0.3, random_state=100)
#print(X_train)
#print(y_train)
df = pd.get_dummies(df,columns = ["Coverage"])
df = pd.get_dummies(df,columns = ["State"])
df = pd.get_dummies(df,columns = ["Education"])
df = pd.get_dummies(df,columns = ["EmploymentStatus"])
df = pd.get_dummies(df,columns = ["Location Code"])
df = pd.get_dummies(df,columns = ["Marital Status"])
df = pd.get_dummies(df,columns = ["Policy Type"])
df = pd.get_dummies(df,columns = ["Renew Offer Type"])
df = pd.get_dummies(df,columns = ["Sales Channel"])
df = pd.get_dummies(df,columns = ["Vehicle Size"])
df.head()

#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X_train,y_train)
y_pred = log.predict(X_test)

conf = confusion_matrix(y_test,y_pred)
print(conf)

print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test,y_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X1_train,y1_train)
y1_pred = log.predict(X1_test)

conf = confusion_matrix(y1_test,y1_pred)
print(conf)

print(classification_report(y1_test, y1_pred))

acc = accuracy_score(y1_test,y1_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X2_train,y2_train)
y2_pred = log.predict(X2_test)

conf = confusion_matrix(y2_test,y2_pred)
print(conf)

print(classification_report(y2_test, y2_pred))

acc = accuracy_score(y2_test,y2_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X3_train,y3_train)
y3_pred = log.predict(X3_test)

conf = confusion_matrix(y3_test,y3_pred)
print(conf)

print(classification_report(y3_test, y3_pred))

acc = accuracy_score(y3_test,y3_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X4_train,y4_train)
y4_pred = log.predict(X4_test)

conf = confusion_matrix(y4_test,y4_pred)
print(conf)

print(classification_report(y4_test, y4_pred))

acc = accuracy_score(y4_test,y4_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X5_train,y5_train)
y5_pred = log.predict(X5_test)

conf = confusion_matrix(y5_test,y5_pred)
print(conf)

print(classification_report(y5_test, y5_pred))

acc = accuracy_score(y5_test,y5_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X6_train,y6_train)
y6_pred = log.predict(X6_test)

conf = confusion_matrix(y6_test,y6_pred)
print(conf)

print(classification_report(y6_test, y6_pred))

acc = accuracy_score(y6_test,y6_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X7_train,y7_train)
y7_pred = log.predict(X7_test)

conf = confusion_matrix(y7_test,y7_pred)
print(conf)

print(classification_report(y7_test, y7_pred))

acc = accuracy_score(y7_test,y7_pred)
print("accuracy = ",acc)
#build the model
log = LogisticRegression(solver='liblinear')
model = log.fit(X8_train,y8_train)
y8_pred = log.predict(X8_test)

conf = confusion_matrix(y8_test,y8_pred)
print(conf)

print(classification_report(y8_test, y8_pred))

acc = accuracy_score(y8_test,y8_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test,y_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X1_train, y1_train)

y1_pred = classifier.predict(X1_test)

print(confusion_matrix(y1_test, y1_pred))
print(classification_report(y1_test, y1_pred))

acc = accuracy_score(y1_test,y1_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X2_train, y2_train)

y2_pred = classifier.predict(X2_test)

print(confusion_matrix(y2_test, y2_pred))
print(classification_report(y2_test, y2_pred))

acc = accuracy_score(y2_test,y2_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X3_train, y3_train)

y3_pred = classifier.predict(X3_test)

print(confusion_matrix(y3_test, y3_pred))
print(classification_report(y3_test, y3_pred))

acc = accuracy_score(y3_test,y3_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X4_train, y4_train)

y4_pred = classifier.predict(X4_test)

print(confusion_matrix(y4_test, y4_pred))
print(classification_report(y4_test, y4_pred))

acc = accuracy_score(y4_test,y4_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X5_train, y5_train)

y5_pred = classifier.predict(X5_test)

print(confusion_matrix(y5_test, y5_pred))
print(classification_report(y5_test, y5_pred))

acc = accuracy_score(y5_test,y5_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X6_train, y6_train)

y_pred = classifier.predict(X6_test)

print(confusion_matrix(y6_test, y6_pred))
print(classification_report(y6_test, y6_pred))

acc = accuracy_score(y6_test,y6_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X7_train, y7_train)

y7_pred = classifier.predict(X7_test)

print(confusion_matrix(y7_test, y7_pred))
print(classification_report(y7_test, y7_pred))

acc = accuracy_score(y7_test,y7_pred)
print("accuracy = ",acc)
classifier = DecisionTreeClassifier()
classifier.fit(X8_train, y8_train)

y8_pred = classifier.predict(X8_test)

print(confusion_matrix(y8_test, y8_pred))
print(classification_report(y8_test, y8_pred))

acc = accuracy_score(y8_test,y8_pred)
print("accuracy = ",acc)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
model = lin_reg.fit(X_train,y_train)

print(f'Coefficients: {lin_reg.coef_}')
print(f'Intercept: {lin_reg.intercept_}')
print(f'R^2 score: {lin_reg.score(X, y)}')
print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')
print(f'R^2 score for test: {lin_reg.score(X_test, y_test)}')

X_sm = X
X_sm = sm.add_constant(X_sm)
lm = sm.OLS(y,X_sm).fit()
lm.summary()