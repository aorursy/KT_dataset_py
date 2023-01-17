import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/heart-disease-uci/heart.csv')

print('The DataFrame has ' + str(df.isnull().sum().sum()) + ' missing values')

df.head()
x = df['thal'].where(df['thal']==0)

x.dropna()
df.shape
corr = df.corr()

df.corr()# Compute the correlation matrix
# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



print('SOME COLUMNS MAY BE IRRELAVANT')

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
gen = df['sex'].where(df['target'] == 1).value_counts().to_frame().transpose() # Total 138

gen.columns = ['Male','Female']

a = gen.plot(kind = 'bar')

plt.xlabel('Gender')

plt.ylabel('Number of Patients')

plt.title('GENDER')

plt.show(a)

age = df['age'].where(df['target'] == 1).value_counts().to_frame()

age = age.reset_index()

b = plt.scatter(y = 'age',x = 'index' ,data = age)

plt.ylabel('Number of Patients')

plt.xlabel('Age')

plt.title('Age Bias')

plt.show(b)
#cp - chest pain type (4 values)(1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)

fig,axs = plt.subplots(1,2,figsize = (20,6))

cp_dat = df['cp'].where(df['target'] == 1).value_counts().to_frame().transpose()

cp_dat_total = df['cp'].value_counts().to_frame().transpose()

cp_dat.rename(columns = {0:'typical angina', 1:'atypical angina', 2:'non-anginal pain', 3:'asymptomatic'},inplace = True)

cp_dat_total.rename(columns = {0:'typical angina', 1:'atypical angina', 2:'non-anginal pain', 3:'asymptomatic'},inplace = True)

cp_dat.plot(kind = 'bar',ax = axs[0],color = ['blue','orange','green','red'])

print('PEOPLE WHO SUFFER AN ATTACK')

print(cp_dat)

cp_dat_total.plot(kind = 'bar',ax = axs[1],color = ['green','blue','orange','red'])

print('PEOPLE WHO WERE SURVEYED')

print(cp_dat_total)
bpdat = df['trestbps'].where(df['target'] == 1).value_counts().to_frame()

bpdat = bpdat.reset_index()

c = plt.scatter(y = 'trestbps',x = 'index' ,data = bpdat)

plt.ylabel('Number of Patients')

plt.xlabel('At Rest Blood Pressure')

plt.title('WRT TO BLOOD PRESSURE AT REST (mm Hg)')

plt.show(c)
choldat = df['chol'].where(df['target'] == 1).value_counts().to_frame()

choldat = choldat.reset_index()

d = plt.scatter(y = 'chol',x = 'index' ,data = choldat)

plt.ylabel('Number of Patients')

plt.xlabel('Cholestrol Level in mg/dl')

plt.title('WRT Serum Cholestoral')

plt.show(d)
fbsdat = df['sex'].where(df['target'] == 1).value_counts().to_frame().transpose() # Total 138

fbsdat.columns = ['>120 mg/dl','<120 mg/dl']

e = fbsdat.plot(kind = 'bar')

plt.xlabel('Fasting Blood Sugar')

plt.ylabel('Number of Patients')

plt.title('Fasting Blood Sugar(in mg/dl)')

plt.show(e)
exangdat = df['exang'].where(df['target'] == 1).value_counts().to_frame().transpose() # Total 138

exangdat.columns = ['Yes','No']

f = exangdat.plot(kind = 'bar')

plt.xlabel('Excercise Induced Angina')

plt.ylabel('Number of Patients')

plt.title('Excercise Induced Angina(Yes or No)')

plt.show(f)
hrdat = df['thalach'].where(df['target'] == 1).value_counts().to_frame()

hrdat = hrdat.reset_index()

g = plt.scatter(y = 'thalach',x = 'index' ,data = hrdat)

plt.ylabel('Number of Patients')

plt.xlabel('Maximum Heart Rate Achieved')

plt.title('WRT Maximum Heart Rate Achieved')

plt.show(g)
#restecg - resting electrocardiographic results (values 0,1,2)(0 = normal; 1 = having ST-T; 2 = hypertrophy)

fig,axs = plt.subplots(1,2,figsize = (20,6))

ecg_dat = df['restecg'].where(df['target'] == 1).value_counts().to_frame().transpose()

ecg_dat_total = df['restecg'].value_counts().to_frame().transpose()

ecg_dat.rename(columns = {0:'normal', 1:'having ST-T', 2:'hypertrophy'},inplace = True)

ecg_dat_total.rename(columns = {0:'normal', 1:'having ST-T', 2:'hypertrophy'},inplace = True)

ecg_dat.plot(kind = 'bar',ax = axs[0])

print('PEOPLE WHO SUFFER AN ATTACK')

print(ecg_dat)

ecg_dat_total.plot(kind = 'bar',ax = axs[1])

print('PEOPLE WHO WERE SURVEYED')

print(ecg_dat_total)
opdat = df['oldpeak'].where(df['target'] == 1).value_counts().to_frame()

opdat = opdat.reset_index()

h = plt.scatter(y = 'oldpeak',x = 'index' ,data = opdat)

plt.ylabel('Number of Patients')

plt.xlabel('Old Peak')

plt.title('ST depression induced by exercise relative to rest')

plt.show(h)
#slope - the slope of the peak exercise ST segment(1 = upsloping; 2 = flat; 3 = downsloping)

fig,axs = plt.subplots(1,2,figsize = (20,6))

slope_dat = df['slope'].where(df['target'] == 1).value_counts().to_frame().transpose()

slope_dat_total = df['slope'].value_counts().to_frame().transpose()

slope_dat.rename(columns = {0:'upsloping', 1:'flat', 2:'downsloping'},inplace = True)

slope_dat_total.rename(columns = {0:'upsloping', 1:'flat', 2:'downsloping'},inplace = True)

slope_dat.plot(kind = 'bar',ax = axs[0])

print('PEOPLE WHO SUFFER AN ATTACK')

print(slope_dat)

slope_dat_total.plot(kind = 'bar',ax = axs[1])

print('PEOPLE WHO WERE SURVEYED')

print(slope_dat_total)
#ca - number of major vessels (0-3) colored by flourosopy

fig,axs = plt.subplots(1,2,figsize = (20,6))

ca_dat = df['ca'].where(df['target'] == 1).value_counts().to_frame().transpose()

ca_dat_total = df['ca'].value_counts().to_frame().transpose()

ca_dat.rename(columns = {0:'One', 1:'Two', 2:'Three',3:'Four',4:'Five'},inplace = True)

ca_dat_total.rename(columns = {0:'One', 1:'Two', 2:'Three',3:'Four',4:'Five'},inplace = True)

ca_dat.plot(kind = 'bar',ax = axs[0],color = ['blue','Orange','Green','Red','purple'])

print('PEOPLE WHO SUFFER AN ATTACK')

print(ca_dat)

ca_dat_total.plot(kind = 'bar',ax = axs[1],color = ['blue','Orange','Green','purple','Red'])

print('PEOPLE WHO WERE SURVEYED')

print(ca_dat_total)
X = np.asarray(df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach','exang','oldpeak','slope','ca','thal']])

y = np.asarray(df['target'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C = 0.5, solver='newton-cg').fit(X_train,y_train)
yhat = LR.predict(X_test)

print(yhat)

yhat_prob = LR.predict_proba(X_test)

print(yhat_prob)
from sklearn.metrics import jaccard_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss

print('The Log Loss is: ' + str(log_loss(y_test, yhat_prob)))

print('Jaccard Similarity Score is : ' + str(jaccard_score(y_test, yhat)))

print('F1 Score is : ' + str(f1_score(y_test,yhat)))
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(LR, X_test, y_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, yhat))