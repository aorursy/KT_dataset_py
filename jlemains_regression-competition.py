import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import chardet
from datetime import datetime
sns.set(rc={'figure.figsize':(35,15)})
matplotlib.rc('figure', figsize=(15, 7))
matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 
matplotlib.rc('axes', titlesize=17)
pd.options.mode.chained_assignment = None  # default='warn'
data = pd.read_csv('../input/train.csv')
print("Taille de la base de données : ",data.shape)
data.head()
ind = ['SalePrice', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
       'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
description = [None, ' The building class', ' The general zoning classification',
       ' Linear feet of street connected to property',
       ' Lot size in square feet', ' Type of road access',
       ' Type of alley access', ' General shape of property',
       ' Flatness of the property', ' Type of utilities available',
       ' Lot configuration', ' Slope of property',
       ' Physical locations within Ames city limits',
       ' Proximity to main road or railroad',
       ' Proximity to main road or railroad (if a second is present)',
       ' Type of dwelling', ' Style of dwelling',
       ' Overall material and finish quality',
       ' Overall condition rating', ' Original construction date',
       ' Remodel date', ' Type of roof', ' Roof material',
       ' Exterior covering on house',
       ' Exterior covering on house (if more than one material)',
       ' Masonry veneer type', ' Masonry veneer area in square feet',
       ' Exterior material quality',
       ' Present condition of the material on the exterior',
       ' Type of foundation', ' Height of the basement',
       ' General condition of the basement',
       ' Walkout or garden level basement walls',
       ' Quality of basement finished area',
       ' Type 1 finished square feet',
       ' Quality of second finished area (if present)',
       ' Type 2 finished square feet',
       ' Unfinished square feet of basement area',
       ' Total square feet of basement area', ' Type of heating',
       ' Heating quality and condition', ' Central air conditioning',
       ' Electrical system', ' First Floor square feet',
       ' Second floor square feet',
       ' Low quality finished square feet (all floors)',
       ' Above grade (ground) living area square feet',
       ' Basement full bathrooms', ' Basement half bathrooms',
       ' Full bathrooms above grade', ' Half baths above grade',
       ' Number of bedrooms above basement level', ' Number of kitchens',
       ' Kitchen quality',
       ' Total rooms above grade (does not include bathrooms)',
       ' Home functionality rating', ' Number of fireplaces',
       ' Fireplace quality', ' Garage location', ' Year garage was built',
       ' Interior finish of the garage',
       ' Size of garage in car capacity',
       ' Size of garage in square feet', ' Garage quality',
       ' Garage condition', ' Paved driveway',
       ' Wood deck area in square feet',
       ' Open porch area in square feet',
       ' Enclosed porch area in square feet',
       ' Three season porch area in square feet',
       ' Screen porch area in square feet', ' Pool area in square feet',
       ' Pool quality', ' Fence quality',
       ' Miscellaneous feature not covered in other categories',
       ' $Value of miscellaneous feature', ' Month Sold', ' Year Sold',
       ' Type of sale', ' Condition of sale']
dicti = pd.DataFrame(data = description, columns = [0], index = ind)
dicti.head()
data.SalePrice.hist(bins=70)
plt.title('Distribution des prix')
plt.show()
data.SalePrice.describe()
data = data.drop(data[data.SalePrice>np.percentile(data.SalePrice, 99)].index).reset_index()
print("Nouvelle taille : ",data.shape)
data.SalePrice.hist(bins=70)
plt.title('Distribution des prix')
plt.show()
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn import preprocessing
import sklearn.linear_model 
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
import copy  
# Affichage d'un tableau avec le pourcentage des valeurs manquantes et le type de chaque variables
def print_MV_percentage(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()*100/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total de VM', 'Pourcentage'])
    missing_data = missing_data[missing_data['Total de VM']!=0]

    long=[]
    for i in range (0, len(missing_data.index)):
        long.append((data[missing_data.index[i]]).dtype)
    missing_data['Type'] = long
    return missing_data

# Méthode qui remplie les valeurs manquantes grace à l'algorithme de KNN (uniquement pour les variables numériques)
def fill_MV_KNN(data):
    data_num = data.select_dtypes(exclude=['object'])
    for col in data.columns:
        neighR = KNeighborsRegressor(n_neighbors=2)
        if data[col].isnull().sum() != 0:
            #subset of columns with no missing values
            columns_no_nan = data_num.dropna(axis=1, how='any').columns
            # X to predict
            X_pred = data[data[col].isnull()][columns_no_nan]
            # X for training
            X = data[columns_no_nan]
            # y with no MV
            y_full = data[col].dropna()
            #index of no missing values
            index = y_full.index
            #fit known rows
            found=False
            if is_numeric_dtype(data[col]):
                neighR.fit(X.loc[index],y_full)
                pred = neighR.predict(X_pred)
                found=True
            if found != False: 
                #create data frame with the prediction
                df_pred = pd.DataFrame(data = pred,index = X_pred.index,columns = ['value'])
                #Fill:
                for i in df_pred.index:
                    data.at[i,col]=df_pred.at[i,'value']

# Affichage de 4/2 distributions de variables
def four_subplots(data,A,B,C,D):
    plt.subplot(2,2,1) 
    if is_numeric_dtype(data[A]):
        plt.boxplot(data[A], 0, 'rs', 0)
    else : 
        data[A].value_counts(dropna=False).plot(kind='bar')
    text = dicti.at[A,0] + ' VM: ' + str(data[A].isnull().sum())
    plt.title(text)
    
    plt.subplot(2,2,2)            
    if is_numeric_dtype(data[B]):
        plt.boxplot(data[B], 0, 'rs', 0)
    else : 
        data[B].value_counts(dropna=False).plot(kind='bar')
    text = dicti.at[B,0] + ' VM: ' + str(data[B].isnull().sum())
    plt.title(text)
    
    plt.subplot(2,2,3)            
    if is_numeric_dtype(data[C]):
        plt.boxplot(data[C], 0, 'rs', 0)
    else : 
        data[C].value_counts(dropna=False).plot(kind='bar')
    text = dicti.at[C,0] + ' VM: ' + str(data[C].isnull().sum())
    plt.title(text)
    
    plt.subplot(2,2,4)            
    if is_numeric_dtype(data[D]):
        plt.boxplot(data[D], 0, 'rs', 0)
    else : 
        data[D].value_counts(dropna=False).plot(kind='bar')
    text = dicti.at[D,0] + ' VM: ' + str(data[D].isnull().sum())
    plt.title(text)
    plt.tight_layout()
    plt.show()
def two_subplots(data,A,B):
    plt.subplot(1,2,1) 
    if is_numeric_dtype(data[A]):
        plt.boxplot(data[A], 0, 'rs', 0)
    else : 
        data[A].value_counts(dropna=False).plot(kind='bar')
    text = dicti.at[A,0] + ' VM: ' + str(data[A].isnull().sum())
    plt.title(text)
    
    plt.subplot(1,2,2)            
    if is_numeric_dtype(data[B]):
        plt.boxplot(data[B], 0, 'rs', 0)
    else : 
        data[B].value_counts(dropna=False).plot(kind='bar')
    text = dicti.at[B,0] + ' VM: ' + str(data[B].isnull().sum())
    plt.title(text)

    plt.tight_layout()
    plt.show()

# Affichage de 4/2 distributions des variables en fonction du prix
def four_subplots_price(data,A,B,C,D):
    plt.subplot(2,2,1) 
    plt.scatter(data[A],data.SalePrice)
    text = dicti.at[A,0] + ' VM: ' + str(data[A].isnull().sum())
    plt.title(text)
    plt.xticks(rotation=90)
    
    plt.subplot(2,2,2)            
    plt.scatter(data[B],data.SalePrice)
    text = dicti.at[B,0] + ' VM: ' + str(data[B].isnull().sum())
    plt.title(text)
    plt.xticks(rotation=90)
    
    plt.subplot(2,2,3)            
    plt.scatter(data[C],data.SalePrice)
    text = dicti.at[C,0] + ' VM: ' + str(data[C].isnull().sum())
    plt.title(text)
    plt.xticks(rotation=90)
    
    plt.subplot(2,2,4)            
    plt.scatter(data[D],data.SalePrice)
    text = dicti.at[D,0] + ' VM: ' + str(data[D].isnull().sum())
    plt.title(text)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()   
def two_subplots_price(data,A,B):
    plt.subplot(1,2,1) 
    plt.scatter(data[A],data.SalePrice)
    text = dicti.at[A,0] + ' VM: ' + str(data[A].isnull().sum())
    plt.title(text)
    plt.xticks(rotation=90)
    
    plt.subplot(1,2,2)            
    plt.scatter(data[B],data.SalePrice)
    text = dicti.at[B,0] + ' VM: ' + str(data[B].isnull().sum())
    plt.title(text)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Création d'une nouvelle colonne à partir d'une liste de colonne (en faisant la moyenne des valeurs des colonnes de la liste)
def create_new_col(data,A,listi,noplot) :
    #Création d'une variable (somme normalisé de toutes les variables)
    data[A] = data[listi].sum(axis=1).apply(lambda x : x/len(listi))
    if ('SalePrice' in data.columns) and (noplot==False):
        plt.scatter(data[A],data.SalePrice)
        title = "New column : " + A
        plt.title(title)

# Renvoie différents score lorsqu'on applique l'algorithme KBest avec l'algo de ML mis en argument
def score_kbest(data,model,name):
    list_score = []
    list_mae = []
    list_rmse = []

    #The smaller the means squared error, the closer you are to finding the line of best fit.

    x = data.drop(['SalePrice'],axis=1)
    y = data.SalePrice           
    
    for i in range(1,x.shape[1]+1):
              
        X_new = SelectKBest(chi2, k=i).fit_transform(x,y)
        xtrain, xtest, ytrain, ytest = train_test_split(X_new, y,random_state=0)

        model.fit(xtrain,ytrain)
        pred = model.predict(xtest)
        list_score.append(model.score(xtest,ytest))
        list_mae.append(median_absolute_error(pred,ytest))
        list_rmse.append(mean_squared_error(np.log(pred),np.log(ytest)))

    plt.plot(range(1,x.shape[1]+1),list_rmse, 'o-',label=name)
    plt.xlabel("Nombre de features selectionnés",fontsize=15)
    plt.ylabel("Score",fontsize=15)
    plt.legend()
    
    return [[name,max(list_score),list_score.index(max(list_score))+1,
             min(list_mae),list_mae.index(min(list_mae))+1,
             min(list_rmse),list_rmse.index(min(list_rmse))+1,]]

# Affichage de la variation de l'erreur en fonction du k dans KNN
def plot_KNN_errors(x,y):
    from sklearn.neighbors import KNeighborsRegressor
    xtrain, xtest, ytrain, ytest = train_test_split(x, y,random_state=0)
    errors=[]
    for k in range (1,25):
        knn = KNeighborsRegressor(k)
        errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest,ytest)))
    plt.xlabel("valeur de k")
    plt.ylabel("valeur de l'erreur")
    plt.plot(range(1,25),errors, 'o-')
    plt.show()

# Encodage des données
def encoding(data):
    #colonnes catégorielles
    df = copy.deepcopy(data)
    for i in df.select_dtypes(include=['object']).columns:
        list_unique = set(df[i].unique())
        dict_pro = dict(zip(list_unique,np.arange(len(list_unique))))
        df[i] = df[i].map(dict_pro)
    return df

# Normalisation des données grace au MinMaxScaler
def normalize(data,listi):
    data_drop = copy.deepcopy(data)
               
    min_max_scaler = preprocessing.MinMaxScaler()
    listi.remove('SalePrice')
    
    data_drop[listi]=min_max_scaler.fit_transform(data_drop[listi])
            
    return data_drop

# Evaluation de la qualité en fonction de la liste de qualité donnée
def eval_quality(x,list_qual):    
    df_score = pd.DataFrame(data = list(reversed(np.arange(len(list_qual)))))
    df_score.index = list_qual
    return df_score.at[x,0]

# Application d'une fonction qui évalue la qualité pour la convertir en note
def eval_data_quality(data):
    list_qual = ['Ex','Gd','TA','Av','Fa','Mn','Po','No','None']
    for i in data.columns:
        inter = set(data[i].unique()).intersection(list_qual)
        if bool(inter) & (inter != {"None"}):
            data[i] = data[i].apply(lambda x: eval_quality(x,list_qual))
    return data

# Affichage de l'importance des variable grace à un arbre ExtraTrees
def feature_importance (data,target,text):
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(random_state=0)
    model.fit(data,target)
    importance = model.feature_importances_

    sorted_feature = []
    indices = np.argsort(importance)[::-1]
    for f in range(data.shape[1]):
        classi = "%d. feature %s (%f)" % (f + 1, data.columns[indices[f]], importance[indices[f]])
        print(classi)
        sorted_feature.append(data.columns[indices[f]]) 
    
    # Plot the feature importances of the forest
    fig,ax = plt.subplots()
    plt.title("Feature importances with ExtraTreesClassifier",fontsize=15)
    if text==1:
        for i, v in enumerate(importance[indices]):
            ax.text(i - 0.15, v + 0.005, round(v,2),fontsize=15)
    
    plt.bar(range(data.shape[1]), importance[indices], color="r", align="center")
    plt.xticks(range(data.shape[1]), sorted_feature,rotation=90,fontsize=15)
    plt.xlim([-1, data.shape[1]])
    plt.show()

# Renvoie les scores d'un ExtraTreesClassifier
def score_extra(data,model,name):
    
    list_score = []
    list_mae = []
    list_rmse = []

    x = data.drop(['SalePrice'],axis=1)
    y = data.SalePrice    
    
    tree = ExtraTreesRegressor(random_state=0)
    tree.fit(x,y)
    importance = tree.feature_importances_
    
    sorted_feature = []
    indices = np.argsort(importance)[::-1]
    
    for f in range(x.shape[1]):
        sorted_feature.append(x.columns[indices[f]])  
   
    for i in range(1,x.shape[1]+1):
        
        
        X_new = x[sorted_feature[1:(i+1)]]
        xtrain, xtest, ytrain, ytest = train_test_split(X_new, y,random_state=0)

        model.fit(xtrain,ytrain)
        pred = model.predict(xtest)
        list_score.append(model.score(xtest,ytest))
        list_mae.append(median_absolute_error(pred,ytest))
        list_rmse.append(mean_squared_error(np.log(abs(pred)),np.log(abs(ytest))))

    plt.plot(range(1,x.shape[1]+1),list_rmse, 'o-',label=name)
    plt.xlabel("Nombre de features selectionnés",fontsize=15)
    plt.ylabel("Score",fontsize=15)
    plt.legend()
    
    return [[name,max(list_score),list_score.index(max(list_score))+1,
             min(list_mae),list_mae.index(min(list_mae))+1,
             min(list_rmse),list_rmse.index(min(list_rmse))+1,]]

# Evaluation des paramètres grace au GridSearch
def eval_param(model,data):
    alphas = np.arange(0.1,1,0.01)
    # create and fit a ridge regression model, testing each alpha
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(data.drop(['SalePrice'],axis=1),data.SalePrice)
    
    # summarize the results of the grid search
    print("Score de : ",grid.best_score_," avec alpha = ",grid.best_estimator_.alpha)

# Evaluation des différents algorithmes avec la méthode kbest
def try_regression_models(data_test,k):
    tableau_score = pd.DataFrame()

    ridge = linear_model.Ridge(alpha=1)
    tableau_score = tableau_score.append(score_kbest(data_test,ridge,"Ridge"))

    regression = linear_model.LinearRegression()
    tableau_score = tableau_score.append(score_kbest(data_test,regression,"Linear Regression"))

    lasso = linear_model.Lasso(alpha=1)
    tableau_score = tableau_score.append(score_kbest(data_test,lasso,"LASSO"))

    neigh = KNeighborsRegressor(n_neighbors=k)
    tableau_score = tableau_score.append(score_kbest(data_test,neigh,"KNN"))

    regr = linear_model.ElasticNet(random_state=0,alpha = 1)
    tableau_score = tableau_score.append(score_kbest(data_test,regr,"ElasticNet"))

    rf = RandomForestRegressor(random_state=0)
    tableau_score = tableau_score.append(score_kbest(data_test,rf,"RF"))

    grad = GradientBoostingRegressor()
    tableau_score = tableau_score.append(score_kbest(data_test,grad,"Gradient Boosting"))

    tableau_score.columns = ['Model','R2','K (R2)','MAE','K (MAE)','RMSE','K (RMSE)']
    
    plt.title("Variation du score des régressions en fonction du nombre de variables choisies (KBest)")
    
    return tableau_score.reset_index(drop=True).sort_values(by=['RMSE'], ascending=True)

# Evaluation des différents algorithmes avec la méthode ExtraTreesClassifier
def try_regression_models_trees(data_test,k):
    tableau_score = pd.DataFrame()

    ridge = linear_model.Ridge(alpha=1)
    tableau_score = tableau_score.append(score_extra(data_test,ridge,"Ridge"))

    regression = linear_model.LinearRegression()
    tableau_score = tableau_score.append(score_extra(data_test,regression,"Linear Regression"))

    lasso = linear_model.Lasso(alpha=1)
    tableau_score = tableau_score.append(score_extra(data_test,lasso,"LASSO"))

    neigh = KNeighborsRegressor(n_neighbors=k)
    tableau_score = tableau_score.append(score_extra(data_test,neigh,"KNN"))

    regr = linear_model.ElasticNet(random_state=0,alpha = 0.01)
    tableau_score = tableau_score.append(score_extra(data_test,regr,"ElasticNet"))

    rf = RandomForestRegressor(random_state=0)
    tableau_score = tableau_score.append(score_extra(data_test,rf,"RF"))

    grad = GradientBoostingRegressor()
    tableau_score = tableau_score.append(score_extra(data_test,grad,"Gradient Boosting"))

    tableau_score.columns = ['Model','R2','K (R2)','MAE','K (MAE)','RMSE','K (RMSE)']
    
    plt.title("Variation du score des régressions en fonction du nombre de variables choisies (Trees)")
    
    return tableau_score.reset_index(drop=True).sort_values(by=['RMSE'], ascending=True)
drop_cols = []
print(data.select_dtypes(exclude=['object']).columns)
print("Len",len(data.select_dtypes(exclude=['object']).columns))
# Variable sans interet car 90% des données ont la meme valeur
drop_cols.append('Id')
#On calcule le pourcentage de valeur manquantes pour chaque features et on les trie.
data_num = data.select_dtypes(exclude=['object'])
print_MV_percentage(data_num)
#Only GarageYrBlt & MasVnrArea is necessary to fill with a special value because there is no garage
for i in range(data.shape[0]):
    data.at[i,'GarageYrBlt'] = data['GarageYrBlt'].fillna(value=0).values[i]
    data.at[i,'MasVnrArea'] = data['MasVnrArea'].fillna(value=0).values[i]
from sklearn.neighbors import KNeighborsRegressor
fill_MV_KNN(data)
data_num = data.select_dtypes(exclude=['object'])
print_MV_percentage(data_num)
#Fonction qui enlève les outliers (superieur au 99% percentile)
def remove_outliers(data,list_col):
    for col in list_col:
        data = data.drop(data[(data[col]>np.percentile(data[col], 99)) & (data.SalePrice<400000)].index)
    return data
# Remove outliers
list_outliers = ['LotFrontage','LotArea','1stFlrSF','GrLivArea','BedroomAbvGr','TotRmsAbvGrd','OpenPorchSF','3SsnPorch','MiscVal']
data = remove_outliers(data,list_outliers)
four_subplots(data,'MSSubClass','LotFrontage', 'LotArea','OverallQual')
four_subplots_price(data,'MSSubClass','LotFrontage', 'LotArea','OverallQual')
((data.OverallQual + data.OverallCond)/2).hist()
four_subplots(data,'OverallCond', 'YearBuilt', 'YearRemodAdd','MasVnrArea')
four_subplots_price(data,'OverallCond', 'YearBuilt', 'YearRemodAdd','MasVnrArea')
four_subplots(data,'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')
four_subplots_price(data,'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF')
four_subplots(data,'1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea')
four_subplots_price(data,'1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea')
four_subplots(data,'BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath')
four_subplots_price(data,'BsmtFullBath', 'BsmtHalfBath', 'FullBath','HalfBath')
four_subplots(data,'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces')
four_subplots_price(data,'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces')
four_subplots(data,'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF')
four_subplots_price(data,'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF')
four_subplots(data,'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch')
four_subplots_price(data,'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch')
four_subplots(data,'PoolArea','MiscVal', 'MoSold', 'YrSold')
four_subplots_price(data,'PoolArea','MiscVal', 'MoSold', 'YrSold')
print(data.select_dtypes(include=['object']).columns)
print("Len",len(data.select_dtypes(include=['object']).columns))
#On calcule le pourcentage de valeur manquantes pour chaque features et on les trie.
data_o = data.select_dtypes(include=['object'])
print_MV_percentage(data_o)
#Some features needs special fill
data['Electrical']=data['Electrical'].dropna()
#Dans tous les autres cas, les valeurs manquantes sont présentes lorsque la maison n'a pas cette option, 
#on remplace donc la valeur manquante par "None"
data = data.fillna(value="None")
data_o = data.select_dtypes(include=['object'])
print_MV_percentage(data_o)
four_subplots(data,'MSZoning', 'Street', 'Alley', 'LotShape')
# Variable sans interet car 90% des données ont la meme valeur
drop_cols.append('Street')
four_subplots_price(data,'MSZoning', 'Street', 'Alley', 'LotShape')
four_subplots(data,'LandContour', 'Utilities','LotConfig', 'LandSlope')
# Variable sans interet car 90% des données ont la meme valeur
drop_cols.append('Utilities')
four_subplots_price(data,'LandContour', 'Utilities','LotConfig', 'LandSlope')
four_subplots(data,'Neighborhood', 'Condition1', 'Condition2','BldgType')
# Variable sans interet car 90% des données ont la meme valeur
drop_cols.append('Condition2')
four_subplots_price(data,'Neighborhood', 'Condition1', 'Condition2','BldgType')
four_subplots(data,'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st')
four_subplots_price(data,'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st')
four_subplots(data,'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond')
four_subplots_price(data,'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond')
four_subplots(data,'Foundation','BsmtQual', 'BsmtCond', 'BsmtExposure')
four_subplots_price(data,'Foundation','BsmtQual', 'BsmtCond', 'BsmtExposure')
four_subplots(data,'CentralAir', 'Electrical', 'KitchenQual','Functional')
four_subplots_price(data,'CentralAir', 'Electrical', 'KitchenQual','Functional')
four_subplots(data,'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual')
four_subplots_price(data,'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual')
four_subplots(data,'GarageCond', 'PavedDrive', 'PoolQC', 'Fence')
four_subplots_price(data,'GarageCond', 'PavedDrive', 'PoolQC', 'Fence')
four_subplots(data,'MiscFeature','SaleType', 'SaleCondition','SaleCondition')
data.MiscFeature.value_counts()
four_subplots_price(data,'MiscFeature','SaleType', 'SaleCondition','SaleCondition')
#Colonne concernant les pièces de la maison
cols_rooms = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Bedroom','TotRmsAbvGrd','Kitchen','1stFlrSF']
four_subplots_price(data,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath')
#Creation d'une nouvelle colonne qui est une combinaison linéaire des 4 présentes dans la liste
#Cette création permet de réduire le nombre de colonne
create_new_col(data,'Bath',['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],False)
four_subplots_price(data,'BedroomAbvGr','TotRmsAbvGrd','KitchenAbvGr','1stFlrSF')
create_new_col(data,'Rooms',['BedroomAbvGr','TotRmsAbvGrd','KitchenAbvGr'],False)

#Conversion des qualité en note
data = eval_data_quality(data)
#Colonnes qui désignent des qualités
cols_quality = ['OverallQual','OverallCond','ExterQual','ExterCond','BsmtQual','BsmtCond',
                'BsmtExposure','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
four_subplots_price(data,'OverallQual','OverallCond','ExterQual','ExterCond')
create_new_col(data,'Overall',['OverallQual','OverallCond'],False)
four_subplots_price(data,'BsmtQual','BsmtCond','BsmtExposure','KitchenQual')
create_new_col(data,'BsmtScore',['BsmtQual','BsmtCond','BsmtExposure'],False)
four_subplots_price(data,'FireplaceQu','GarageQual','GarageCond','PoolQC')
create_new_col(data,'GarScore',['GarageQual','GarageCond'],False)
create_new_col(data,'ExterScore',['ExterQual','ExterCond'],False)
create_new_col(data,'OtherScore',['PoolQC','KitchenQual','FireplaceQu'],False)
data.Neighborhood.value_counts()
def apply_site_info(data):
    #Liste créée à la main avec les notes de chaque voisinnage (dispo sur le site)
    rate = [11,7,1,4,8,10,4,5,11,10,10,10,6,11,11,11,9,4,5,8,8,11,11,10,11]
    vois = ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr','Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel','NAmes','NoRidge','NPkVill','NridgHt','NWAmes','OldTown','SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker']
    dico = dict(zip(vois,rate))
    data['Like_Neigh'] = data.Neighborhood.map(dico)
    rate = [11,3,11,10,1,8,10,5,11,8,8,8,9,11,11,11,6,10,5,1,1,11,11,6,11]
    dico = dict(zip(vois,rate))
    data['Exp_Neigh'] = data.Neighborhood.map(dico)
apply_site_info(data)
plt.scatter(data.Like_Neigh,data.SalePrice)
plt.scatter(data.Exp_Neigh,data.SalePrice)
#Creation d'une colonne qui désigne le prix au m2 de la maison
array = data.SalePrice/data.LotArea
array = array.rename('Price/feet')
data_price = pd.concat([data.Neighborhood,data.MSZoning,array],axis=1)
data_price.head()
#Fonction qui fait la moyenne de tous les prix au m2 calculés pour trouver le prix au m2 du voisinnage
def fill_neighborhood_price(data,data_price):
    data_vois = pd.DataFrame(columns = ['Neighborhood','Price'])
    i = 0
    for vois in data.Neighborhood.unique():
        data_vois.at[i,'Neighborhood'] = vois
        data_vois.at[i,'Price'] = data_price.loc[data_price.Neighborhood == vois,'Price/feet'].mean()
        i = i+1
    for vois in data_vois.Neighborhood:
        data.loc[data.Neighborhood == vois,'Price/neigh'] = data_vois.loc[data_vois.Neighborhood == vois,'Price'].values

#Fonction qui fait la moyenne de tous les prix au m2 calculés pour trouver le prix au m2 de la zone
def fill_zone_price(data,data_price):
    data_zone = pd.DataFrame(columns = ['Zone','Price'])
    i = 0
    for zone in data.MSZoning.unique():
        data_zone.at[i,'Zone'] = zone
        data_zone.at[i,'Price'] = data_price.loc[data_price.MSZoning == zone,'Price/feet'].mean()
        i = i+1
    
    for zone in data_zone.Zone:
        data.loc[data.MSZoning == zone,'Price/zone'] = data_zone.loc[data_zone.Zone == zone,'Price'].values
fill_neighborhood_price(data,data_price)
fill_zone_price(data,data_price)
data.Neighborhood.value_counts().plot(kind='barh')
plt.xticks(rotation=90)
plt.tight_layout()
plt.title('Prix au m2 en fonction du voisinage')
plt.show()
data.YrSold.value_counts()
list_price_year = []
for year in data.YrSold.unique():
    mean = data.loc[data.YrSold == year,'SalePrice'].mean()
    list_price_year.append(mean)
plt.title("Moyenne des prix en fonction de l'année de vente")
plt.bar(data.YrSold.unique(),list_price_year)
plt.show()
serie = data['SalePrice'].groupby([data['YrSold'],data['MoSold']]).mean()
serie.plot()
plt.xticks(range(0,len(serie.index.ravel())),serie.index.ravel(),rotation=90)
plt.show()
#Colonne qui me semblait influencer le prix d'une maison 
cols = ['MSZoning','LotArea','Neighborhood','ExterScore','YearRemodAdd','Like_Neigh','Exp_Neigh','GarageYrBlt', 'GarageCars',
        'GarageArea','OtherScore','GarScore','BsmtScore','MiscVal','Rooms','1stFlrSF','Bath',
        'YearBuilt','YrSold','MoSold','LotFrontage','SalePrice','TotalBsmtSF']
data_select = data[cols]
data_select.head()
data_select.head()
def normalize_year(data):
    listi = (data.YearRemodAdd/data.YearBuilt.min()).values
    data['YearRemod'] = listi
    listi = (data.YearBuilt/data.YearBuilt.min()).values
    data['YearB'] = listi
    listi = (data.YrSold/data.YearBuilt.min()).values
    data['YearSold'] = listi
    listi = (data.GarageYrBlt/data.YearBuilt.min()).values
    data['GarageYr'] = listi
    data = data.drop(['YearRemodAdd','YearBuilt','YrSold','GarageYrBlt','MSZoning'],axis=1)
    return data
data_select = normalize_year(data_select)
data_select.head()
data_encode = encoding(data_select)
data_encode.head()
data_one = pd.get_dummies(data_select)
data_one.head()
data_encode = data_encode.reset_index(drop=True)
data_one = data_one.reset_index(drop=True)
cols_to_keep = ['TotalBsmtSF','Bath','Rooms','YearB','GarageCars','OtherScore']
list_to_scale = list(data_encode.columns)
list_to_scale = [col for col in list_to_scale if col not in cols_to_keep]
data_test = normalize(data_encode,list_to_scale)
data_test.describe().T
cols_to_keep = ['Neighborhood','TotalBsmtSF','Bath','Rooms','YearB','GarageCars','OtherScore']
list_to_scale = list(data_encode.columns)
list_to_scale = [col for col in list_to_scale if col not in cols_to_keep]
data_one_norm = normalize(data_one,list_to_scale)
data_one_norm.head()
x = data_test.drop(['SalePrice'],axis =1)
y = data_test.SalePrice
feature_importance (x,y,0)      
x_one = data_one_norm.drop(['SalePrice'],axis =1)
y_one = data_one_norm.SalePrice
feature_importance (x_one,y_one,0)
selector = SelectKBest (chi2, k= 'all')
X_new = selector.fit_transform(x,y)
names = x.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]

names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
ns_df_sorted.head(10)
selector = SelectKBest (chi2, k= 'all')
X_new = selector.fit_transform(x_one,y_one)
names = x_one.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]

names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
ns_df_sorted.head(10)
from sklearn import linear_model
print('Pour Ridge :')
model = linear_model.Ridge()
eval_param(model,data_test)
print('Pour Lasso :')
model = linear_model.Lasso()
eval_param(model,data_test)
print('Pour ElasticNet :')
model = linear_model.ElasticNet()
eval_param(model,data_test)
plot_KNN_errors(data_test.drop(['SalePrice'],axis=1),data_test.SalePrice)
try_regression_models(data_test,5)
try_regression_models_trees(data_test,5)
1- 12000/181000
grad = GradientBoostingRegressor()

x = data_test.drop(['SalePrice'],axis=1)
y = data_test.SalePrice    
    
tree = ExtraTreesRegressor(random_state=0)
tree.fit(x,y)
importance = tree.feature_importances_
    
sorted_feature = []
indices = np.argsort(importance)[::-1]
    
for f in range(x.shape[1]-1):
    sorted_feature.append(x.columns[indices[f]])

X_new = x[sorted_feature[1:21]]
xtrain, xtest, ytrain, ytest = train_test_split(X_new, y,random_state=0)

grad.fit(xtrain,ytrain)
pred = grad.predict(xtest)

columns_to_keep = X_new.columns

print("RMSE score :",mean_squared_error(np.log(abs(pred)),np.log(abs(ytest))))
plot_KNN_errors(data_one_norm.drop(['SalePrice'],axis=1),data_one_norm.SalePrice)
try_regression_models(data_one_norm,5)
try_regression_models_trees(data_one_norm,5)
test = pd.read_csv('../input/test.csv')
ind = test.Id
test.shape
data_num = test.select_dtypes(exclude=['object'])
print_MV_percentage(data_num)
test[data_num.columns] = test[data_num.columns].fillna(0)
test.MSZoning = test.MSZoning.fillna(test.MSZoning.value_counts().index[0])
test = test.fillna('None')
# Rooms
create_new_col(test,'Bath',['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],True)
create_new_col(test,'Rooms',['BedroomAbvGr','TotRmsAbvGrd','KitchenAbvGr'],True)
#Quality
test = eval_data_quality(test)
create_new_col(test,'Overall',['OverallQual','OverallCond'],True)
create_new_col(test,'BsmtScore',['BsmtQual','BsmtCond','BsmtExposure'],True)
create_new_col(test,'GarScore',['GarageQual','GarageCond'],True)
create_new_col(test,'ExterScore',['ExterQual','ExterCond'],True)
create_new_col(test,'OtherScore',['PoolQC','KitchenQual','FireplaceQu'],True)
#Site
apply_site_info(test)
cols.remove('SalePrice')
#Selection
test = test[cols]
fill_neighborhood_price(test,data_price)
fill_zone_price(test,data_price)
test['Zone'] = test.MSZoning.map({'RL':0,'RM':1,'C (all)':1,'None':1,'FV':1,'RH':2})
test.Zone.value_counts()
test = normalize_year(test)
list_to_scale = list(test.columns)
list_to_scale = [col for col in list_to_scale if col not in ['TotalBsmtSF','Bath','Rooms','YearB','GarageCars','OtherScore']]
list_to_scale.append('SalePrice')
list_to_scale.remove('Neighborhood')
test_one = normalize(pd.get_dummies(test).reset_index(drop=True),list_to_scale)
test_one.shape
test = encoding(test)
list_to_scale = list(test.columns)
list_to_scale = [col for col in list_to_scale if col not in ['TotalBsmtSF','Bath','Rooms','YearB','GarageCars','OtherScore']]
list_to_scale.append('SalePrice')
test = normalize(test.reset_index(drop=True),list_to_scale)
test = test[columns_to_keep]
prediction = grad.predict(test)
test_sub_int = pd.DataFrame(data = ind , columns = ['Id','SalePrice'])
test_sub_int['SalePrice'] = prediction
test_sub_int.to_csv('submission_int.csv', index=False)
#0.21
data = eval_data_quality(data)
four_subplots_price(data,'OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch')
create_new_col(data,'Porch',['OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch'],False)
drop_cols.extend(['OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch'])
four_subplots_price(data,'BsmtQual','BsmtCond', 'BsmtExposure','BsmtFinType1')
four_subplots_price(data,'BsmtFinSF1','BsmtFinType1', 'BsmtFinSF2','BsmtFinType2')
two_subplots_price(data,'BsmtFullBath','BsmtHalfBath')
#Same as TotalBsmtSF
drop_cols.extend(['BsmtQual','BsmtCond', 'BsmtExposure'])
plt.scatter(data.TotalBsmtSF,data.SalePrice)
four_subplots_price(data,'Exterior1st','Exterior2nd', 'ExterQual','ExterCond')
#Exterior1st ~Exterior2nd
drop_cols.extend(['ExterQual','ExterCond','Exterior2nd'])
two_subplots_price(data,'PoolArea','PoolQC')
data['Pool'] = data.PoolArea.apply(lambda x : 1 if x>0 else 0)
drop_cols.extend(['PoolArea','PoolQC'])
drop_cols
data = data.drop(data[drop_cols],axis=1)
data = data.drop(['index'],axis=1)
data.shape
data_encode = encoding(data)
data.head()
data_encode.head()
data_one = pd.get_dummies(data)
data_one.head()
x = data_encode.drop(['SalePrice'],axis=1)
y = data_encode.SalePrice
feature_importance (x,y,0)
xo = data_one.drop(['SalePrice'],axis=1)
yo = data_one.SalePrice

feature_importance (xo,yo,0)
plot_KNN_errors(x,y)
try_regression_models(data_encode,4)
try_regression_models_trees(data_encode,4)
grad_all = GradientBoostingRegressor()

x = data_encode.drop(['SalePrice'],axis=1)
y = data_encode.SalePrice    

selector = SelectKBest(chi2, k=60)
X_new = selector.fit_transform(x,y)
xtrain, xtest, ytrain, ytest = train_test_split(X_new, y,random_state=0)

grad_all.fit(xtrain,ytrain)
pred = grad_all.predict(xtest)

columns_to_keep = selector.get_support(indices=True)

print("RMSE score :",mean_squared_error(np.log(abs(pred)),np.log(abs(ytest))))
test = pd.read_csv('../input/test.csv')
test_num = test.select_dtypes(exclude=['object'])
print_MV_percentage(test_num)
#Only GarageYrBlt & MasVnrArea is necessary to fill with a special value because there is no garage
for i in range(test.shape[0]):
    test.at[i,'GarageYrBlt'] = test['GarageYrBlt'].fillna(value=0).values[i]
    test.at[i,'MasVnrArea'] = test['MasVnrArea'].fillna(value=0).values[i]

from sklearn.neighbors import KNeighborsRegressor
fill_MV_KNN(test)
test_num = test.select_dtypes(exclude=['object'])
print_MV_percentage(test_num)
test_num = test.select_dtypes(include=['object'])
print_MV_percentage(test_num)
for col in ['MSZoning','Utilities','Functional','KitchenQual','SaleType','Exterior2nd','Exterior1st']:
    test[col] = test[col].fillna(test[col].value_counts().index[0])   
test = test.fillna(value="None")
test_num = test.select_dtypes(include=['object'])
print_MV_percentage(test_num)
test = eval_data_quality(test)
create_new_col(test,'Porch',['OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch'],True)
create_new_col(test,'Bath',['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],True)
create_new_col(test,'Rooms',['BedroomAbvGr','TotRmsAbvGrd','KitchenAbvGr'],True)
create_new_col(test,'Overall',['OverallQual','OverallCond'],True)
create_new_col(test,'BsmtScore',['BsmtQual','BsmtCond','BsmtExposure'],True)
create_new_col(test,'GarScore',['GarageQual','GarageCond'],True)
create_new_col(test,'ExterScore',['ExterQual','ExterCond'],True)
create_new_col(test,'OtherScore',['PoolQC','KitchenQual','FireplaceQu'],True)
test['Pool'] = test.PoolArea.apply(lambda x : 1 if x>0 else 0)
apply_site_info(test)
drop_cols.remove('Id')
fill_neighborhood_price(test,data_price)
fill_zone_price(test,data_price)
test = test.drop(test[drop_cols],axis=1)
id_ = test.Id
test = test.drop(['Id'],axis=1)
test = encoding(test)
test = test[data_encode.columns[columns_to_keep]]
prediction = grad_all.predict(test)
test_sub = pd.DataFrame(data = id_ , columns = ['Id','SalePrice'])
test_sub['SalePrice'] = prediction
test_sub.to_csv('submission.csv', index=False)
#0.16
