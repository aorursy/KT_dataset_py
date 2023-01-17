!pip install pycaret
from pycaret.datasets import get_data

diabetes = get_data('diabetes')
from pycaret.classification import *

exp1 = setup(diabetes,target = 'Class variable',silent=True)
compare_models()
adaboost = create_model('ada')
ridge_classifier = create_model('ridge')
adaboost.base_estimator_
adaboost.feature_importances_
tuned_adaboost = tune_model(adaboost)
dec_tree = create_model('dt')
dec_tree_bagged = ensemble_model(dec_tree)
plot_model(adaboost,plot='auc')
plot_model(adaboost,plot='boundary')
plot_model(adaboost,plot='pr')
plot_model(adaboost,plot='vc')
plot_model(adaboost,plot='confusion_matrix')
plot_model(adaboost,plot='learning')
evaluate_model(adaboost)
rf = create_model('rf')
rf_holdout_test = predict_model(rf)
predictions = predict_model(rf,data=diabetes)
predictions
deploy_model(model=rf,model_name='rf_aws',platform='aws',authentication={

    'bucket':'pycaret-test'

})
save_model(rf,model_name='rf_fo_deployment')