# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# coding: utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


FEATURES = set()
AGE_FEATURES = set()
FARE_TABLE = None
AGE_PRED_MODEL = None
DF_FEATURE_MAX = pd.DataFrame()
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    } 

def fea_title(df):
	df['Title'] = df['Name'].apply(lambda x: Title_Dictionary[x.split(',')[1].split('.')[0].strip()])
	return dummy_feature(df, 'Title')

'''def fea_ticket(df):
	def Ticket_Prefix(s):
		s=s.split()[0]
		return 'NoClue' if s.isdigit() else s
	df['TicketPrefix'] = df['Ticket'].apply(lambda x: Ticket_Prefix(x))
	#sns.countplot(x="TicketPrefix", data=df, hue='Survived')
	#plt.show()
	return dummy_feature(df, 'TicketPrefix')'''

def fea_embarked(df):
	df['Embarked']=df.apply(lambda x: 'S' if pd.isnull(x['Embarked']) else x['Embarked'], axis=1)
	return dummy_feature(df, 'Embarked')

def fea_pclass(df):
	return dummy_feature(df, 'Pclass')

def fea_sex(df):
	global FEATURES
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
	FEATURES |= set(['Gender'])
	return df

def fea_family(df):
	global FEATURES
	df['Family'] = df['SibSp'] + df['Parch'] + 1
	FEATURES |= set(['Family'])
	return df

def fea_family_size(df):
	global FEATURES
	df['Singleton'] = (df['Family'] == 1).astype(int)
	df['FamilySmall'] = np.logical_and(df['Family'] > 1, df['Family'] < 5).astype(int)
	df['FamilyLarge'] = (df['Family'] >= 5).astype(int)
	FEATURES |= set(['Singleton', 'FamilySmall', 'FamilyLarge'])
	return df

def fea_fare(df):
	global FEATURES, FARE_TABLE
	FARE_TABLE=df.pivot_table(values='Fare', columns=['Pclass','Sex','Embarked'], aggfunc='mean')
	FEATURES |= set(['Fare'])
	return df

def fea_fare_fill(df):
	global FARE_TABLE
	df['Fare'] = df[['Fare', 'Pclass', 'Sex', 'Embarked']].apply(lambda x:
                            FARE_TABLE[x['Pclass']][x['Sex']][x['Embarked']] if pd.isnull(x['Fare'])
                            else x['Fare'], axis=1)
	return df

def fea_age(df):
	global FEATURES, AGE_PRED_MODEL, AGE_FEATURES

	from sklearn.svm import SVR
	from sklearn.grid_search import GridSearchCV

	features = list(FEATURES)
	AGE_FEATURES = FEATURES.copy()
	df_use = df[['Age'] + features]
	X_age = df_use.dropna()[features].as_matrix()[:,:]
	y_age = df_use.dropna()['Age'].as_matrix().astype(float)
	df_pred = df_use[df_use['Age'].isnull()]
	X_age_pred = df_pred[features].as_matrix()[:,:]

	svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5, n_jobs=-1, \
	                   param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5), "epsilon":np.linspace(.1,.5,5)})
	svr.fit(X_age, y_age)

	df['Age']=df_use.apply(lambda x: svr.predict(x[features].reshape(1, -1)) if pd.isnull(x['Age']) else x['Age'], axis=1)
	FEATURES |= set(['Age'])
	AGE_PRED_MODEL=svr
	return df

def fea_age_fill(df):
	global AGE_PRED_MODEL
	age_features = list(AGE_FEATURES)
	df_use = df[['Age'] + age_features]
	df['Age']=df_use.apply(lambda x: AGE_PRED_MODEL.predict(x[age_features].reshape(1, -1)) if pd.isnull(x['Age']) else x['Age'], axis=1)
	return df

def fea_child(df):
	global FEATURES
	df['Child'] = (df['Age'] < 10).astype(int)
	FEATURES |= set(['Child'])
	return df

def dummy_feature(df, column, dummy_na=False):
	global FEATURES
	df_dum = pd.get_dummies(df[column], prefix=column, dummy_na=dummy_na).applymap(np.int)
	FEATURES |= set(df_dum.columns)
	return pd.concat([df,df_dum], axis=1)

def feature_engineering(df, funcs):
	for func in funcs:
		df=func.__call__(df)
	return df

def scale_all_features(df):
	global DF_FEATURE_MAX, FEATURES
	features = list(FEATURES)
	DF_FEATURE_MAX = df[features].max()
	df[features] = df[features].apply(lambda x: x/x.max(), axis=0)
	return df

def scale_test_features(df):
	global DF_FEATURE_MAX, FEATURES
	features = list(FEATURES)
	df[features] /= DF_FEATURE_MAX
	return df

def get_cross_val_score(X, y):
	clf = RandomForestClassifier(n_estimators=100, max_features=.8, max_depth=7, n_jobs=-1)
	# 5-fold cross-validation
	scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
	return scores

def train_grid_search(X, y):
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.grid_search import GridSearchCV

	est_range = list(range(10, 30, 2)) + list(range(30, 150, 10))
	fea_range = np.arange(.5,.8,.1).tolist()
	depth_range = np.arange(5.,10.,1.).tolist() + [None]
	'''est_range = list(xrange(100, 120, 10))
				fea_range = np.arange(.5,.7,.1).tolist()
				depth_range = np.arange(5,7,1).tolist() + [None]'''
	parameter_grid = {
	    'n_estimators': est_range,
	    'max_features': fea_range,
	    'max_depth': depth_range
	}
	gs = GridSearchCV(RandomForestClassifier(n_estimators = 10), parameter_grid, n_jobs=-1, cv=5, verbose=3, scoring='accuracy')
	gs.fit(X,y)
	return gs.best_estimator_, gs.best_params_

def plot_feature_importance(model, feature_names, max_nums=10):
	import matplotlib.pyplot as plt
	nums = len(feature_names) if len(feature_names) < max_nums else max_nums
	feature_importance = model.feature_importances_
	# make importances relative to max importance
	feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance)
	sorted_idx = sorted_idx[:nums]
	pos = np.arange(sorted_idx.shape[0]) + .5
	plt.subplot(1, 1, 1)
	plt.barh(pos, feature_importance[sorted_idx], align='center')

	feature_labels = pd.Series(feature_names)
	plt.yticks(pos, feature_labels[sorted_idx])
	plt.xlabel('Relative Importance')
	plt.title('Variable Importance')
	plt.show()	

def pred(model):
	global FARE_TABLE,AGE_PRED_MODEL, FEATURES
	features = list(FEATURES)
	df_pred = pd.read_csv('../input/test.csv')
	fea_funcs = [fea_title, fea_embarked, fea_pclass, fea_sex, fea_family, fea_family_size,\
		 fea_fare_fill, fea_age_fill, fea_child]
	df_fea = feature_engineering(df_pred, fea_funcs)
	X = df_fea[features].as_matrix()[:,:]
	output = model.predict(X)

	ids = df_pred['PassengerId'].astype(int)
	result = np.c_[ids, output.astype(int)]
	df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])
	df_result.to_csv('04_1.csv', index=False)


def train():
	global FEATURES
	FEATURES = set()
	FARE_TABLE = None
	AGE_PRED_MODEL = None
	df = pd.read_csv('../input/train.csv')
	fea_funcs = [fea_embarked, fea_pclass, fea_sex, fea_family, fea_family_size, fea_fare, fea_age, fea_child]
	df_fea = feature_engineering(df, fea_funcs)

	features = list(FEATURES)
	X = df_fea[features].as_matrix()[:,:]
	y = df_fea['Survived'].as_matrix().astype(int)

	model, params = train_grid_search(X,y)
	plot_feature_importance(model, features)
	return model

def get_cross_val_statics_score(X, y, nums=1):
	scores = []
	for i in range(nums):
		scores += list(get_cross_val_score(X, y))
	return np.mean(scores), np.std(scores)

def cross_val(funcs, nums=1):
	global FEATURES, FARE_TABLE, AGE_PRED_MODEL
	FEATURES = set()
	FARE_TABLE = None
	AGE_PRED_MODEL = None
	df = pd.read_csv('../input/train.csv')
	df_fea = feature_engineering(df, funcs)

	# set is not hashable for dataframe
	features = list(FEATURES)
	X = df_fea[features].as_matrix()[:,:]
	y = df_fea['Survived'].as_matrix().astype(int)

	score_mean, score_std = get_cross_val_statics_score(X, y, nums)
	print("score mean = %s,std = %s" %(score_mean, score_std))
	return score_mean, score_std

def plot_feature_curve(feature_labels, scores_mean, scores_std, title='feature score curve', ylim=None):
    """
    生成feature处理的curve

    Parameters
    ----------
    feature_labels : string
    	处理feature的标签

    scores_mean : array-like, shape (n_samples)
    	成绩means

    scores_std : array-like, shape (n_samples)
        stds
    """
    plt.figure()
    plt.title(title)

    plt.xlabel("feature engineering")
    plt.ylabel("Score")
    plt.grid()

    arr_mean = np.array(scores_mean, dtype=np.float64)
    arr_std = np.array(scores_std, dtype=np.float64)

	# 数值ticket替换
    x_ticket_proxy = np.arange(len(feature_labels))
    plt.fill_between(x_ticket_proxy, arr_mean - arr_std, arr_mean + arr_std, alpha=0.1, color="g")
    plt.plot(x_ticket_proxy, arr_mean, 'o-', color="g", label="Cross-validation score")
    axes = plt.gca()
    axes.set_xlim([-1,np.max(x_ticket_proxy)+1])
    axes.set_ylim([np.min(arr_mean - arr_std*2),np.max(arr_mean + arr_std*2)])
    plt.xticks(x_ticket_proxy, feature_labels)
    #plt.legend(loc="best")
    plt.show()

def test_feas():
	# child 和 familysize相关
	feature_labels = []
	feature_funcs = [fea_embarked, fea_pclass, fea_sex, fea_fare]
	feature_funcs_adds = [fea_age, fea_child, fea_family, fea_family_size, fea_title]
	scores_mean = []
	scores_std = []

	for fea_func in feature_funcs_adds:
		feature_funcs.append(fea_func)
		feature_labels.append(fea_func.__name__)
		score_mean, score_std = cross_val(feature_funcs, 10)
		scores_mean.append(score_mean)
		scores_std.append(score_std)
	plot_feature_curve(feature_labels, scores_mean, scores_std)

if __name__ == '__main__': 
	#test_feas()
	#model = train()
	#pred(model)
	pass

test_feas()
model = train()
pred(model)
