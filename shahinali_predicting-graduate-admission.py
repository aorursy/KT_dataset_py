import pandas as pd
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df_test = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
print(df_train.count()) 

print(df_test.count())
df_train.columns
df_train.drop('Serial No.', axis=1, inplace=True)
df_test.drop('Serial No.', axis=1, inplace=True)

df_train.head()
df_train['admission'] =  np.where(df_train['Chance of Admit '] >= 0.75, 1, 0)
df_train.head()
import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (8, 6)
# plt.rcParams['font.size'] = 14
# Pandas scatter plot
df_train.plot(kind='scatter', x='GRE Score', y='CGPA')
sns.scatterplot(x='GRE Score', y='CGPA', hue="admission", data=df_train)
feature_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ', 'CGPA', 'Research']

# multiple scatter plots in Seaborn
g = sns.pairplot(df_train, x_vars=feature_cols, y_vars='Chance of Admit ', kind='reg')

for chts in g.axes[0]: 
    chts.axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')

sns.relplot(x='GRE Score', y='CGPA',
                 col="University Rating", hue="admission", 
                 kind="scatter", data=df_train)
sns.relplot(x='GRE Score', y='TOEFL Score',
                 col="University Rating", hue='admission', 
                 kind="scatter", data=df_train)
pdf=df_train.groupby(['Research','University Rating']).mean().reset_index()
pdf
bg = sns.boxplot(y="Chance of Admit ",  x= 'Research', palette=["m", "g"], data=df_train)
sns.despine(offset=10, trim=True)
bg.axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')
bg = sns.boxplot(y="Chance of Admit ",  x= 'University Rating', data=df_train)
sns.despine(offset=10, trim=True)
bg.axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')
gr = sns.catplot(x = "University Rating",   
            y = "Chance of Admit ",       
            hue = "Research",  
            data = df_train.groupby(['Research','University Rating']).mean().reset_index() , 
            kind = "bar")
gr.axes[0][0].axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')
sns.relplot(x='CGPA', y='Research',
            col="University Rating", hue='admission', 
            kind="scatter", data=df_train)
colormap = sns.diverging_palette(100, 5, as_cmap=True)
sns.heatmap(df_train.corr(), annot = True, cmap= colormap, cbar=True,  fmt=".2f" )
corr = df_train.corr()
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(100, 5, as_cmap=True)

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(corr,cmap=colormap,linewidths=.5, annot=True, mask=dropSelf )
# import, instantiate, fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split


linreg = LinearRegression()
feature_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']

## training set
X = df_train[feature_cols]
y = df_train['Chance of Admit ']

## test set 
X_test = df_test[feature_cols]
y_test = df_test['Chance of Admit ']

## fit model
linreg.fit(X, y)
# print the coefficients
print(list(zip(feature_cols,linreg.coef_)))
# define a function that accepts a list of features and returns RMSE, prediction 
def train_test_rmse(feature_cols, X , y):
    y_pred = linreg.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred)), y_pred
rmse, ypred = train_test_rmse(feature_cols, X_test , y_test)

df_test['admission_predict'] = y_pred
print(rmse)
print('MAE:',  mean_absolute_error(y_test, y_pred), ' ',  (1./len(y_test))*(sum(abs(y_test-y_pred))))
print('MSE:', mean_squared_error(y_test, y_pred), ' ',   (1./len(y_test))*(sum((y_test-y_pred)**2)))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)), ' ', sqrt((1./len(y_test))*(sum((y_test-y_pred)**2))))
fig, ax = plt.subplots()
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(7, 7)})
sns.regplot(x=y_test, y=y_pred,  scatter=False, ax=ax);
##Check for Linearity
f = plt.figure(figsize=(14,5))
## linear
ax = f.add_subplot(121)
sns.scatterplot(y_test,y_pred,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual error
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.distplot((y_test-y_pred), bins = 50)
ax.axvline((y_test - y_pred).mean(),color='r',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror');
sns.distplot(y_test,hist=True,label = 'Actual')
sns.distplot(y_pred,hist=True, label ='Predicted')
plt.legend(loc="upper right")
plt.xlabel('Prediction')
from sklearn.preprocessing import StandardScaler as SS
ss = SS()
X_ss = ss.fit_transform(X)
ss1 = SS()
X_test_ss = ss.fit_transform(X_test)
linreg.fit(X_ss, y)
 
rmse_ss = train_test_rmse(feature_cols, X_test_ss , y_test)[0]
y_pred_ss = train_test_rmse(feature_cols, X_test_ss , y_test)[1]
print(rmse_ss)
sns.distplot(y_test,hist=True,label = 'Actual')
sns.distplot(y_pred_ss,hist=True, label ='Predicted (for StandardScaler inputs)')
plt.legend(loc="upper right")
plt.xlabel('Prediction')