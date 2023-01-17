!pip install pycaret
from pycaret.datasets import get_data
data=get_data('anomaly')
from pycaret.anomaly import *
setup=setup(data)
iforest=create_model('iforest')
plot_model(iforest)
knn=create_model('knn')
plot_model(knn)
knn_prediction=predict_model(knn, data=data)
knn_prediction