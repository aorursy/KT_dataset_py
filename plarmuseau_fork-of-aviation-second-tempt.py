import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
#del train,test
#train=pd.read_csv("../input/train.csv")
#test=pd.read_csv("../input/test.csv")


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def kodeer(dfdir):
    
    mtrain=pd.read_csv(dfdir).reset_index()
    if 'experiment' in mtrain.columns:
        print('train experiment /n',mtrain.groupby('experiment').count())
    else:
        print('event',mtrain.groupby('event').count())
    
                
    features_n = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]

    mtrain['pilot'] = 100 * mtrain['seat'] + mtrain['crew']
    print("Number of pilots : ", len(mtrain['pilot'].unique()))
    
    pilots=mtrain['pilot'].unique()
    for pilot in pilots:
        ids = mtrain[mtrain["pilot"] == pilot].index
        scaler = MinMaxScaler()
        mtrain.loc[ids, features_n] = scaler.fit_transform(mtrain.loc[ids, features_n])    
    
    mtrain=mtrain.drop('experiment',axis=1)
    
    if 'event' in mtrain.columns:
        lbl = LabelEncoder()
        mtrain['event']=lbl.fit_transform(list(mtrain['event'].values))
        lblevent=lbl
        print( list(lblevent.classes_) )
    else:
        lblevent=[]

    mtrain['groep']=np.round( mtrain.index.values/256,0 )  # 256 measurements per second = grouping per second !
    trainSD=mtrain.groupby(['groep','crew']).std()  #,'seat'
    trainMA=mtrain.groupby(['groep','crew']).mean() #,'seat'
    trainSD=trainSD.reset_index().sort_values(['crew','index']) #,'seat'
    trainMA=trainMA.reset_index().sort_values(['crew','index']) #,'seat'

    return trainSD,trainMA,lblevent,mtrain[['groep','crew']] #,'seat'

testsd,testma,lbl_event,test=kodeer("../input/test.csv")
trainsd,trainma,lbl_event,train=kodeer("../input/train.csv")


print(trainsd.shape,testsd.shape,test.shape)

def SVD_tree_predict(e_,mtrain,mtest,veld,idvld):
    velden=[v for v in e_.columns if v not in [veld,idvld]]
    label = mtrain[veld].astype('int')
    mtrain[veld]=label
    print(e_.shape,velden)
    e_=e_.loc[:,velden]

    print(e_.shape)
    ncomp=e_.shape[1]-3
    # SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=ncomp, n_iter=7, random_state=42)
    e_=svd.fit_transform(e_.fillna(0))
    print(e_[:len(mtrain)].shape,mtrain[veld].values.shape)
    
    xtrain=pd.DataFrame(e_[:len(mtrain)])
    xtrain[veld]=label
    xtest=pd.DataFrame(e_[len(mtrain):])
    #New_features=e_[:len(mtrain)]
    #Test_features=e_[len(mtrain):]
    pd.DataFrame(e_[:len(mtrain)]).plot.scatter(x=0,y=1,c=label,colormap='winter')
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier,ExtraTreesClassifier,GradientBoostingRegressor, AdaBoostClassifier
    from sklearn.multiclass import OneVsRestClassifier
    clf=OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10))
    
    fit=clf.fit(e_[:len(mtrain)],label)
    pred=fit.predict(e_[:len(mtrain)])
    from sklearn.metrics import accuracy_score
    print('accuracy',accuracy_score(mtrain[veld].astype('int'),pred)*100)
    #predict
    sub = pd.DataFrame(fit.predict_proba(e_[len(mtrain):]))
    for ci in mtest.columns:
        sub[ci]=mtest[ci]
    
    sub.to_csv('submission.csv', index=False)
    # prepare second method
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, log_loss

    train_df, val_df = train_test_split(mtrain, test_size=0.2, random_state=420)
    print(f"Training on {train_df.shape[0]} samples.")    
    features_n = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]
    features = ["crew", "seat"] + features_n
    features =[x for x in mtrain.columns if x !=veld]  
    def run_lgb(df_train, df_test):
    
        params = {"objective" : "multiclass",
                  "num_class": 4,
                  "metric" : "multi_error",
                  "num_leaves" : 30,
                  "min_child_weight" : 50,
                  "learning_rate" : 0.1,
                  "bagging_fraction" : 0.7,
                  "feature_fraction" : 0.7,
                  "bagging_seed" : 420,
                  "verbosity" : -1
                 }
    
        lg_train = lgb.Dataset(df_train[features], label=(df_train[veld]))
        lg_test = lgb.Dataset(df_test[features], label=(df_test[veld]))
        model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=50, verbose_eval=100)
    
        return model
    
    model = run_lgb(train_df, val_df)
    pred_val = model.predict(val_df[features], num_iteration=model.best_iteration)
    print( confusion_matrix(np.argmax(pred_val, axis=1), val_df[veld].values) )
    pred_test = model.predict(mtest[features], num_iteration=model.best_iteration) #mtest[features]
    submission = pd.DataFrame(np.concatenate((np.arange(len(mtest))[:, np.newaxis], pred_test), axis=1), columns=['id', 'A', 'B', 'C', 'D'])
    submission['id'] = submission['id'].astype(int)
    
    return sub,submission
trainMS=trainma.merge(trainsd,left_index=True,right_index=True)
testMS=testma.merge(testsd,left_index=True,right_index=True)
trainma['event']=trainma['event']+0.249 #*2
trainma['event']=trainma['event'].map(round)

trainma.groupby('event').count(),trainma.shape,testma.shape,trainMS.shape,testMS.shape
trainMS
#subx=SVD_tree_predict(trainMS.append(testMS).drop(['index_x','index_y','id_x','id_y','pilot_x','pilot_y','time_x','time_y','event_y','groep_y','event_x','crew_x','crew_y','seat_x','seat_y'],axis=1), trainma,testma,'event','groep')
dropveld=['index_x','index_y','id_x','id_y','pilot_x','pilot_y','time_x','time_y','event_y','groep_x','groep_y','event_x','crew_x','crew_y','seat_x','seat_y']
dropveld1=['index_x','index_y','pilot_x','pilot_y','time_x','time_y','groep_x','groep_y','crew_x','crew_y','seat_x','seat_y','event_y']

dropveld2=['index_x','index_y','pilot_x','pilot_y','time_x','time_y','groep_x','groep_y','crew_x','crew_y','seat_x','seat_y']

#subx,subg=SVD_tree_predict(trainma.append(testma).drop(['index','id','pilot','time','groep','event','crew','seat'],axis=1), trainma,testma,'event','groep')
subx,subg=SVD_tree_predict(trainMS.append(testMS).drop(dropveld,axis=1), trainMS.drop(dropveld1,axis=1),testMS.drop(dropveld2,axis=1),'event_x','groep')

subx #[['groep','crew','seat']]
subg
sub2=subx.iloc[:,:7]
sub2.columns=['A','B','C','D','groep','crew','seat']
sub2.head()
#sub2=subx.groupby('groep').median()
#sub2=subx[:int(len(subx)/2)]
test
#testSD=testSD.merge(subx,left_on='groep',right_on='groep')
#testSD['event']=lblevent.inverse_transform(testSD['event_x'])


#test=pd.read_csv("../input/test.csv")
#test['groep']=np.round( test.index.values/200,0 )
#test2=test.merge(sub2,how='left',left_on=['groep','crew'],right_on=['groep','crew']) #','seat'
test2=test.merge(subg,how='left',left_on=['groep'],right_on=['id']) #','seat'
#test2.groupby('event_x').count()
#test2=test2.drop(['groep','crew','seat'],axis=1)  #,'seat'
test2=test2.drop(['groep','id','crew'],axis=1)  #,'seat'
test2.index.names = ['id']
test2=test2.reset_index()

test2
test2.to_csv('submission.csv', index=False)