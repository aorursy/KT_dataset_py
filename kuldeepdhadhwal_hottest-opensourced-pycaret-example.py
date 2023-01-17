!pip install pycaret
from pycaret.datasets import get_data
dataset = get_data('credit')
#check the shape of data
dataset.shape
data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'default', session_id=123) 
compare_models()
dt = create_model('dt')
#trained model object is stored in the variable 'dt'. 
print(dt)
knn = create_model('knn')
rf = create_model('rf')
tuned_dt = tune_model('dt')
#tuned model object is stored in the variable 'tuned_dt'. 
print(tuned_dt)
tuned_knn = tune_model('knn')

tuned_rf = tune_model('rf')
plot_model(tuned_rf, plot = 'auc')

plot_model(tuned_rf, plot = 'pr')
plot_model(tuned_rf, plot='feature')

plot_model(tuned_rf, plot = 'confusion_matrix')
predict_model(tuned_rf);
final_rf = finalize_model(tuned_rf)
#Final Random Forest model parameters for deployment
print(final_rf)
predict_model(final_rf);
unseen_predictions = predict_model(final_rf, data=data_unseen)
unseen_predictions.head()

save_model(final_rf,'Final RF Model 20April2020')
saved_final_rf = load_model('Final RF Model 20April2020')
new_prediction = predict_model(saved_final_rf, data=data_unseen)
new_prediction.head()