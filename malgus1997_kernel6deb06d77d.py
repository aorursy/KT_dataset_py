import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statistics
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_validate
from yellowbrick.regressor import ResidualsPlot
from sklearn.decomposition import PCA
%matplotlib inline
data = pd.read_csv('../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv')
data.shape
data.head()
data.shape
data.describe(include = "O")
data_describe = data.describe(percentiles = [.01,.05,.25,.75,.95,.99])
data_describe
numeric_feature = ['Price','Postcode','Propertycount','Distance']
for i in numeric_feature:
    q1 = data_describe[i]['25%']
    q3 = data_describe[i]['75%']
    iqr = q3-q1
    oulier1 = data[data[i]> q3 + 1.5*iqr].index
    oulier2 = data[data[i] < q1-1.5*iqr].index
    oulier = np.concatenate((oulier1, oulier2), axis=0)
    data.drop(oulier,inplace=True)
data.describe(percentiles = [.01,.99])
data.shape
def classify_date(x):
    if str.__contains__(x, '01/2016') or str.__contains__(x, '02/2016') or str.__contains__(x, '03/2016') or str.__contains__(x, '04/2016') or str.__contains__(x, '05/2016') or str.__contains__(x, '06/2016'):
        return 'fh_2016'
    elif str.__contains__(x, '07/2016') or str.__contains__(x, '08/2016') or str.__contains__(x, '09/2016') or str.__contains__(x, '10/2016') or str.__contains__(x, '11/2016') or str.__contains__(x, '12/2016'):
        return 'lh_2016'
    elif str.__contains__(x, '01/2017') or str.__contains__(x, '02/2016') or str.__contains__(x, '03/2017') or str.__contains__(x, '04/2017') or str.__contains__(x, '05/2017') or str.__contains__(x, '06/2017'):
        return 'fh_2017'
    elif str.__contains__(x, '07/2017') or str.__contains__(x, '08/2017') or str.__contains__(x, '09/2017') or str.__contains__(x, '10/2017') or str.__contains__(x, '11/2017') or str.__contains__(x, '12/2017'):
        return 'lh_2017'
    elif str.__contains__(x, '01/2018') or str.__contains__(x, '02/2018') or str.__contains__(x, '03/2018') or str.__contains__(x, '04/2018') or str.__contains__(x, '05/2018') or str.__contains__(x, '06/2018'):
        return 'hf_2018'
    else:
        return 'lh_2018'
data['Period'] = data.Date.apply(lambda x: classify_date(x))
data_Seller_vc = data.SellerG.value_counts().to_frame()
lev_1 = np.array(data_Seller_vc[data_Seller_vc['SellerG']<500].index)
lev_2 = np.array(data_Seller_vc[(data_Seller_vc['SellerG']<1000) & (data_Seller_vc['SellerG']>500)].index)
lev_3 = np.array(data_Seller_vc[(data_Seller_vc['SellerG']<2000) & (data_Seller_vc['SellerG']>1000)].index)
lev_4 = np.array(data_Seller_vc[(data_Seller_vc['SellerG']<4000) & (data_Seller_vc['SellerG']>2000)].index)
lev_5 = np.array(data_Seller_vc[(data_Seller_vc['SellerG']>4000)].index)
data.loc[data.SellerG.isin(lev_1), 'Type_seller']= 'Seller_lev1'
data.loc[data.SellerG.isin(lev_2), 'Type_seller']= 'Seller_lev2'
data.loc[data.SellerG.isin(lev_3), 'Type_seller']= 'Seller_lev3'
data.loc[data.SellerG.isin(lev_4), 'Type_seller']= 'Seller_lev4'
data.loc[data.SellerG.isin(lev_5), 'Type_seller']= 'Seller_lev'
data.drop('Address',axis = 1,inplace=True)
data.drop('Suburb',axis = 1,inplace=True)
data.drop('SellerG',axis = 1,inplace=True)
data.drop('CouncilArea',axis = 1,inplace=True)
data.drop('Date',axis=1,inplace=True)
data.isnull().sum()
for i in data.columns:
    print(data[i].unique())
data.dropna(axis=0,inplace=True)
data['Price'].describe()
sns.distplot(data['Price'])
print("Skewness: %f" % data['Price'].skew())
corr = data.corr()
sns.heatmap(corr, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200),square=True);

numeric_feature = ['Postcode','Propertycount','Distance']
coefficient = []
p_value = []
for i in numeric_feature:
    x, y = stats.pearsonr(data['Price'],data[i])
    coefficient.append(x)
    p_value.append(y)
d = {
    'name': numeric_feature, 
    'Coefficient': coefficient, 
    'p_val': p_value
    }
df = pd.DataFrame(d)
df
data['Rooms'].value_counts()
labels = ['1','2','3','4','5','6','7','8','9','10','31']
values = []
for i in labels:
    values.append(data.Rooms.value_counts()[int(i)])
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()
data.Price.groupby(data['Rooms']).mean()
sns.boxplot(x="Rooms", y="Price", data=data)
P_P = data[data.Price.notna()]
P_P[P_P['Rooms']==1].Price
stats.f_oneway(P_P[P_P['Rooms']==1].Price.values,P_P[P_P['Rooms']==2].Price.values,P_P[P_P['Rooms']==3].Price.values,P_P[P_P['Rooms']==4].Price.values,P_P[P_P['Rooms']==5].Price.values)
data['Type'].value_counts()
labels1 = ['h','u','t']
values1 = []
for i in labels1:
    values.append(data.Type.value_counts()[i])
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()
data.Price.groupby(data['Type']).mean()
sns.distplot(data[data['Type']=='h'].Price, hist = True, kde = True)
sns.distplot(data[data['Type']=='u'].Price, hist = True, kde = True)
sns.distplot(data[data['Type']=='t'].Price, hist = True, kde = True)
stats.f_oneway(P_P[P_P['Type']=='h'].Price,P_P[P_P['Type']=='u'].Price,P_P[P_P['Type']=='t'].Price)
data=data[['Type', 'Method', 'Regionname','Period','Type_seller','Propertycount', 'Distance', 'Postcode','Rooms', 'Price']]
data
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]
data.shape
X['Rooms'] = X['Rooms'].astype('object')
X = pd.get_dummies(X)
#ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(),[0,1,2,3,4])],remainder='passthrough',sparse_threshold=0)
#X = np.array(ct.fit_transform(X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 ,random_state=1)
scaler = MinMaxScaler()
X_train.iloc[:,:4] = scaler.fit_transform(X_train.iloc[:, :4])
X_test.iloc[:,:4] = scaler.fit_transform(X_test.iloc[:, :4])
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred_hd = regressor.predict(X_test)
Y_pred_hd = Y_pred_hd.astype(int)
test_result = np.concatenate((Y_pred_hd.reshape(len(Y_pred_hd),1),Y_test.values.reshape(len(Y_test),1)), axis=1)
test_result = pd.DataFrame(data = test_result, columns =['Y_Predict','Y_test'] )
test_result
sns.scatterplot(x="Y_Predict", y="Y_test", data=test_result)

print('Mean Square Error: {}'.format(mean_squared_error(Y_test, Y_pred_hd)))
print('R2 Score: {}'.format(r2_score(Y_test, Y_pred_hd)))
print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(Y_test, Y_pred_hd))))
print('Percentage between RMSE and Y-Mean: {}%'.format(int((np.sqrt(mean_squared_error(Y_test, Y_pred_hd)))/(Y.mean())*100)))
visualizer = ResidualsPlot(regressor, hist=False)
visualizer.fit(X_train, Y_train) 
visualizer.score(X_test, Y_test) 
visualizer.show()  
pca = PCA()
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
PCA(n_components = 20)
X_train_pca = PCA(n_components = 20).fit_transform(X_train)
X_test_pca= PCA(n_components = 20).fit_transform(X_test)
regressor_pca = LinearRegression()
regressor_pca.fit(X_train_pca,Y_train)
Y_pred_hd_pca = regressor_pca.predict(X_test_pca)
Y_pred_hd_pca = Y_pred_hd_pca.astype(int)
test_result_pca = np.concatenate((Y_pred_hd_pca.reshape(len(Y_pred_hd_pca),1),Y_test.values.reshape(len(Y_test),1)), axis=1)
test_result_pca = pd.DataFrame(data = test_result_pca, columns =['Y_Predict','Y_test'] )
test_result_pca
sns.scatterplot(x="Y_Predict", y="Y_test", data=test_result_pca)

print('Mean Square Error: {}'.format(mean_squared_error(Y_test, Y_pred_hd_pca)))
print('R2 Score: {}'.format(r2_score(Y_test, Y_pred_hd_pca)))
print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(Y_test, Y_pred_hd_pca))))
print('Percentage between RMSE and Y-Mean: {}%'.format(int((np.sqrt(mean_squared_error(Y_test, Y_pred_hd_pca)))/(Y.mean())*100)))
k_fold = []
val_score = []
for i in range(2,11):
    scores = cross_val_score(LinearRegression(), X, Y, cv=i)
    print('Cross validation score with {} fold: {}'.format(i, scores.mean()))
    k_fold.append(i)
    val_score.append(scores.mean())
fig = px.line(x=k_fold, y=val_score, labels = {'x':'k_fold','y':'Cross_val_score'})
fig.show()
Y_pred_cv = cross_val_predict(regressor, X, Y, cv=3)
Y_pred_cv = Y_pred_cv.astype(int)
test_result_cv = np.concatenate((Y_pred_cv.reshape(len(Y_pred_cv),1),Y.values.reshape(len(Y),1)), axis=1)
test_result_cv = pd.DataFrame(data = test_result_cv, columns =['Y_predict','Y'] )
test_result_cv
sns.scatterplot(x="Y_predict", y="Y", data=test_result_cv)
print('Mean Square Error: {}'.format(mean_squared_error(Y, Y_pred_cv)))
print('R2 Score: {}'.format(r2_score(Y, Y_pred_cv)))
print('Root Mean Square Error: {}'.format(np.sqrt(mean_squared_error(Y, Y_pred_cv))))
print('Percentage between RMSE and Y-Mean: {}%'.format(int((np.sqrt(mean_squared_error(Y, Y_pred_cv)))/(Y.mean())*100)))
X_train_no_scale, X_test_no_scale, Y_train_no_scale, Y_test_no_scale = train_test_split(X, Y, test_size = 0.2 ,random_state=1)
regressor_no_scale = LinearRegression()
regressor_no_scale.fit(X_train_no_scale,Y_train_no_scale)
coeff_df = pd.DataFrame(regressor_no_scale.coef_, X.columns, columns=['Coefficient'])  
coeff_df