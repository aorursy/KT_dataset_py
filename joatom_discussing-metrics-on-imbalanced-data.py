from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, classification_report

y_true = [0, 0, 1, 1, 1, 1]

y_pred1 = [1, 1, 1, 1, 1, 1]

y_pred2 = [0, 1, 1, 1, 1, 0]



def eval_preds(y_true, y_pred):

    print('True:',y_true)

    print('Pred:',y_pred)

    print('--------------------------')

    print('accuracy:', accuracy_score(y_true, y_pred))

    print('balcanced_accuracy:', balanced_accuracy_score(y_true, y_pred))

    print('roc_auc:', roc_auc_score(y_true, y_pred))

    print('f1:', f1_score(y_true, y_pred))

    print(classification_report(y_true, y_pred))
eval_preds(y_true, y_pred1)
eval_preds(y_true, y_pred2)