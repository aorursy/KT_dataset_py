import pandas as pd

%pylab inline
ods = pd.read_csv('../input/overdoses.csv')

cleanNum = lambda n: int(n.replace(',', ''))

ods['Deaths'] = ods['Deaths'].map(cleanNum)

ods['Population'] = ods['Population'].map(cleanNum)

ods.head()
# deaths per capita?

ods['deaths_per_capita'] = ods['Deaths'] / ods['Population']

ods = ods.sort(['deaths_per_capita'])



ax = plt.subplot(111)

ods.deaths_per_capita.plot(

kind='bar',

figsize=(12,3),

title="Opioid-related deaths per capita",

)

ax.axis('off')

for i, x in enumerate(ods['Abbrev']):

    ax.text(i-1 + 0.7, 0, x, rotation='45')
ps = pd.read_csv('../input/prescriber-info.csv')

# ps.head()
import xgboost as xgb

def cv (alg,X,y):

    metrics = ['auc', 'map']

    xgtrain = xgb.DMatrix(X,y)

    param = alg.get_xgb_params()

    cvresult = xgb.cv(param,

                      xgtrain,

                      num_boost_round=alg.get_params()['n_estimators'],

                      nfold=7,

                      metrics=metrics,

                      early_stopping_rounds=50)

    alg.set_params(n_estimators=cvresult.shape[0])

    #Predict training set:

    alg.fit(X,y,eval_metric=metrics)

    # Show features, rated by fscore

    features = alg.booster().get_fscore()

    feat_imp = pd.Series(features).sort_values(ascending=False)

    feat_imp[:50].plot(kind='bar', title='Feature Importances', figsize=(9,6))

    plt.ylabel('Feature Importance Score')

    # sort for human readability

    import operator

    sorted_features = sorted(features.items(), key=operator.itemgetter(1))

    print('features by importance', sorted_features)

    return features, cvresult
dentists = ps[ps['Specialty'] == 'Dentist']



dentists['Gender'] = pd.get_dummies(dentists['Gender'])



target = 'Opioid.Prescriber'

X = dentists.drop(['NPI', 'Specialty', 'Credentials', 'State', target], 1)

y = dentists[target]

(X.shape, y.shape)
from sklearn.model_selection import train_test_split

alg = xgb.XGBClassifier(

        learning_rate =0.1,

        n_estimators=1000,

        max_depth=4,

        min_child_weight=1,

        gamma=0,

        subsample=0.8,

        colsample_bytree=0.8,

        nthread=4,

        objective="binary:logistic",

        scale_pos_weight=1,

        seed=27) 



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)



features, cvresults = cv(alg,X_train,y_train)

cvresults[-3:]
from sklearn import metrics

pred = alg.predict(X_test)

predprob = alg.predict_proba(X_test)[:,1]

#Print model report:

print("Accuracy : %.4g" % metrics.accuracy_score(y_test, pred))

print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, predprob))
mean_dentists = dentists.groupby('Opioid.Prescriber').mean()

relevant_stats = [mean_dentists[feature] for feature in features]

pd.DataFrame(relevant_stats).plot(kind="bar")
features
non_opioid_features = [ 'State', 'Gender', 'AMOXICILLIN', 'IBUPROFEN', 'AZITHROMYCIN', 'DOXYCYCLINE.HYCLATE', 'CHLORHEXIDINE.GLUCONATE', 'CEPHALEXIN']
from sklearn.preprocessing import LabelEncoder

series = {}

for feature in non_opioid_features:

    series[feature] = dentists[feature]

X = pd.DataFrame(series)

le = LabelEncoder()

le.fit(X['State'])

X['State'] = le.transform(X['State'])

y = dentists[target]

(X.shape, y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=43)



alg = xgb.XGBClassifier(

        learning_rate =0.1,

        n_estimators=1000,

        max_depth=8,

        min_child_weight=1,

        gamma=1,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        scale_pos_weight=1,

        seed=27) 



features, cvresults = cv(alg,X_train,y_train)

#cvresults[-3:]



pred = alg.predict(X_test)

predprob = alg.predict_proba(X_test)[:,1]

#Print model report:

print("Accuracy : %.4g" % metrics.accuracy_score(y_test, pred))

print("AUC Score (test): %f" % metrics.roc_auc_score(y_test, predprob))