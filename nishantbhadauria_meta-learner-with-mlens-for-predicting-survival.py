import pandas as pd
estonia=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
country_wise=estonia.groupby(['Country'],as_index=False)['Survived'].count()
country_wise['Total_passengers']=country_wise['Survived']
import pycountry
import plotly.express as px
import pandas as pd
import plotly.graph_objs as gobj
list_countries = country_wise['Country'].unique().tolist()
d_country_code = {}  # To hold the country names and their ISO
for country in list_countries:
    try:
        country_data = pycountry.countries.search_fuzzy(country)      
        country_code = country_data[0].alpha_3
        d_country_code.update({country: country_code})
    except:
        print('could not add ISO 3 code for ->', country)
        # If could not find country, make ISO code ' '
        d_country_code.update({country: ' '})
for k, v in d_country_code.items():
    country_wise.loc[(country_wise.Country == k), 'iso_alpha'] = v

fig = px.choropleth(data_frame = country_wise,
                    locations= "iso_alpha",
                    color= "Total_passengers",  
                    hover_name="Country",                   
                    color_continuous_scale=px.colors.sequential.Plasma
                    )
fig.show()
age_wise=estonia.groupby(['Age'],as_index=False)['Survived'].count()
age_wise_sum=estonia.groupby(['Age'],as_index=False)['Survived'].sum()
age_wise_per =(age_wise_sum['Survived'])/(age_wise['Survived'])*100
age_wise_per=pd.DataFrame(age_wise_per)
age_wise_per_d=pd.concat([age_wise['Age'],age_wise_per],axis=1)
import plotly.express as px
px.bar(age_wise_per_d,x='Age',y='Survived',title="Age wise survival %")
passenger_wise=estonia.groupby(['Category'],as_index=False)['Survived'].count()
passenger_wise_surv=estonia.groupby(['Category'],as_index=False)['Survived'].sum()
survival_perce_pass_type=(passenger_wise_surv['Survived']/passenger_wise['Survived'])*100
survival_perce_pass_type_df=pd.concat([passenger_wise['Category'],survival_perce_pass_type],axis=1)
survival_perce_pass_type_df
survived=estonia[estonia['Survived']==1]
deceased=estonia[estonia['Survived']==0]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
import seaborn as sns
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
        count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common surnames in deceased')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
text2=deceased['Lastname'].values
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(text2)
plot_10_most_common_words(count_data, count_vectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import numpy as np
import seaborn as sns
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
        count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common surnames in survived')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
text2=survived['Lastname'].values
count_vectorizer = CountVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(text2)
plot_10_most_common_words(count_data, count_vectorizer)
px.histogram(deceased,x="Age",color="Sex",title="Age wise distribution of sex in deceased")
px.histogram(survived,x="Age",color="Sex",title="Age wise distribution of sex in survived")
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
tf = v.fit_transform(estonia['Lastname'])
estonia['tfidf']=tf.toarray()
estonia['tfidf'].describe()
estonia['agebin'] = pd.cut(estonia['Age'].astype(int), 4)
estonia['country_bin']= [1 if x =='Estonia' or x =='Sweden' else 0 for x in estonia['Country']] 
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()
estonia['sex_cat']=encode.fit_transform(estonia['Sex'])
estonia['agecat'] =encode.fit_transform(estonia['agebin'])
estonia['surname_cat']=encode.fit_transform(estonia['Lastname'])
estonia['passcat'] =encode.fit_transform(estonia['Category'])
x=estonia[['country_bin', 'sex_cat','agecat','passcat','tfidf','surname_cat']]
y=estonia['Survived']
!pip install chars2vec
import chars2vec
import sklearn.decomposition
import matplotlib.pyplot as plt
c2v_model = chars2vec.load_model('eng_50')
word_embeddings = c2v_model.vectorize_words(estonia['Lastname'].to_list())
projection_2d = sklearn.decomposition.PCA(n_components=2).fit_transform(word_embeddings)
f = plt.figure(figsize=(8, 6))
for j in range(len(projection_2d)):
    plt.scatter(projection_2d[j, 0], projection_2d[j, 1],
                marker=('$' + estonia['Lastname'][j] + '$'),
                s=500 * len(estonia['Lastname'][j]), label=j)
plt.show()
from imblearn.over_sampling import SMOTE
over = SMOTE(random_state=0)
ov_x,ov_y=over.fit_sample(x, y)
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(ov_x,ov_y,test_size=0.2,random_state=123)
import xgboost
from xgboost import XGBClassifier
xgc=XGBClassifier(scale_pos_weight=2)
model1=xgc.fit(trainx,trainy)
prediction=model1.predict(testx)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(testy,prediction))
print(classification_report(testy,prediction))
xgc.get_booster().get_score(importance_type= "gain")
from xgboost import plot_importance
import matplotlib.pyplot as plt
plot_importance(model1)
plt.show()
import tensorflow as tf
from tensorflow import keras
from keras.layers import LeakyReLU
from keras.layers import Dense,Activation
from keras.layers.normalization import BatchNormalization
opt2=tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.5, beta_2=0.5, epsilon=1e-07, amsgrad=False,
    name='Adam')
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
from keras.models import Sequential
model2 = Sequential()
model2.add(Dense(300,input_dim=(6))),  
model2.add(Activation('selu')),
model2.add(Dense(100,kernel_regularizer=keras.regularizers.l2(0.01))),
model2.add(Activation('selu')),
model2.add(Dense(20,kernel_regularizer=keras.regularizers.l2(0.01))),
model2.add(LeakyReLU(alpha=0.1)),
model2.add(Dense(2))
model2.add(Activation('softmax'))

epochs=100
optimizers=keras.optimizers.SGD(clipvalue=1.0)
def exp_decay(lr0,s):
    def exp_decay_fn(epcohs):
        return lr0*0.1**(epochs/s)
    return exp_decay_fn

exp_decay_fn=exp_decay(lr0=0.1,s=50)
lr_sch=keras.callbacks.LearningRateScheduler(exp_decay_fn)
lr_sch2=keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)
model2.compile(loss="sparse_categorical_crossentropy",optimizer="Nadam",metrics=["accuracy"])
history=model2.fit(trainx,trainy,epochs=100,callbacks=[lr_sch],verbose=0)
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
predict=model2.predict_classes(testx)
print(confusion_matrix(testy, predict))
print(classification_report(testy, predict))
import lightgbm as lgb
train_data=lgb.Dataset(trainx,label=trainy)
param = {'num_leaves':200, 'objective':'binary','max_depth':10,'learning_rate':.01,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']
num_round=100
lgbm=lgb.train(param,train_data,num_round)
ypred2=lgbm.predict(testx)
for i in range(0,len(testx)):
    if ypred2[i]>=.5:
        ypred2[i]=1
    else:
        ypred2[i]=0
confusion_matrix(testy,ypred2)
import tensorflow as tf
from tensorflow import keras
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Dense,Activation
from keras.layers.normalization import BatchNormalization
opt2=tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.5, beta_2=0.5, epsilon=1e-07, amsgrad=False,
    name='Adam')
sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
from keras.models import Sequential
model3 = Sequential()

model3.add(Dense(300,input_dim=(6))),  
model3.add(Activation('selu')),
model3.add(Dropout(0.2)),
model3.add(Dense(100,kernel_regularizer=keras.regularizers.l2(0.01))),
model3.add(Activation('selu')),
model3.add(Dropout(0.2)),
model3.add(Dense(20,kernel_regularizer=keras.regularizers.l2(0.01))),
model3.add(Dropout(0.2)),
model3.add(Activation('selu')),
model3.add(Dense(2))

model3.add(Activation('softmax'))

epochs=100
optimizers=keras.optimizers.SGD(clipvalue=1.0)
def exp_decay(lr0,s):
    def exp_decay_fn(epcohs):
        return lr0*0.1**(epochs/s)
    return exp_decay_fn

exp_decay_fn=exp_decay(lr0=0.1,s=100)
lr_sch=keras.callbacks.LearningRateScheduler(exp_decay_fn)
lr_sch2=keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=5)
model3.compile(loss="sparse_categorical_crossentropy",optimizer="Nadam",metrics=["accuracy"])
history=model3.fit(trainx,trainy,epochs=100,callbacks=[lr_sch],verbose=0)
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
predict=model3.predict_classes(testx)
print(confusion_matrix(testy, predict))
print(classification_report(testy, predict))
from sklearn.svm import SVC
svm=SVC(kernel='linear')
svm.fit(trainx,trainy)
pred_new=svm.predict(testx)
confusion_matrix(testy,pred_new)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlens.ensemble import SuperLearner
from sklearn.metrics import accuracy_score
def get_models():
    models = list()
    models.append(LogisticRegression(solver='liblinear'))
    models.append(DecisionTreeClassifier())
    models.append(SVC(kernel='linear'))
    models.append(GaussianNB())
    models.append(KNeighborsClassifier())
    models.append(AdaBoostClassifier())    
    models.append(BaggingClassifier(n_estimators=100))
    models.append(RandomForestClassifier(n_estimators=100))
    models.append(ExtraTreesClassifier(n_estimators=100))
    models.append(XGBClassifier(scale_pos_weight=2))
    return models
# create the super learner
def get_super_learner(X):
    ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=False, sample_size=len(X))
    models = get_models()
    ensemble.add(models)
    ensemble.add_meta(DecisionTreeClassifier())
    return ensemble
# create the super learner
ensemble = get_super_learner(trainx)
# fit the super learner
ensemble.fit(trainx, trainy)
# summarize base learners
print(ensemble.data)
# make predictions on hold out set
yhat = ensemble.predict(testx)
print('Super Learner: %.3f' % (accuracy_score(testy, yhat) * 100))
print(confusion_matrix(testy,yhat))
print(classification_report(testy,yhat))