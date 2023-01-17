import pandas as pd



def runCVforPatient(patientNo):

    

    data = pd.read_csv('train_' + str(patientNo) + '_features.csv') 

    

    data.sort_values(['outcome', 'segment'], inplace = True)

    data['seq'] = (data.sequence == 6).shift(1).fillna(0).cumsum()

    data.drop(['sequence'], axis=1, inplace=True)

       

    seq_outcome = data[['seq', 'outcome']].drop_duplicates()

    

    skf = StratifiedKFold(seq_outcome['outcome'], n_folds=5, random_state=1)

    AUC = []

    predictions = np.zeros([data.shape[0]])

    for train_index, test_index in skf:

        seq_train = np.array(seq_outcome.iloc[train_index]['seq'])

        seq_test = np.array(seq_outcome.iloc[test_index]['seq'])

        

        train_data = data[data.seq.isin(seq_train)]

        test_data = data[data.seq.isin(seq_test)]

        

        X_train = train_data.drop(['patient', 'segment', 'outcome', 'seq'], axis=1)

        y_train = train_data['outcome']

        X_test = test_data.drop(['patient', 'segment', 'outcome', 'seq'], axis=1)

        y_test = test_data['outcome']

        

        predictions_indexes = np.array(y_test.index)

        

        clf = LogisticRegression(C=20, n_jobs=-1, verbose=0)

        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)

        this_AUC = roc_auc_score(y_test, y_pred[:,1])

        print ("AUC: " + str(this_AUC))

        AUC.append(this_AUC)

        

        predictions[predictions_indexes] = y_pred[:,1]

    

    print (np.mean(AUC), AUC)

    patient_predictions = pd.DataFrame(predictions)

    patient_predictions = patient_predictions.join(data[['outcome']])

    patient_predictions.columns = ['prediction', 'outcome']

    

    return patient_predictions



#Example (I comment it so it does not return errors):



#all_patient_predictions = pd.DataFrame()

#for patientNo in [1, 2, 3]:

#    patient_predictions = runCVforPatient(patientNo)

#    all_patient_predictions = pd.concat([all_patient_predictions, patient_predictions])

#    

#total_AUC = roc_auc_score(all_patient_predictions['outcome'], all_patient_predictions['prediction'])