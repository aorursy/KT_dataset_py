!pip install pycaret
!pip install flask==0.12.2
!pip install flask-ngrok
from flask import Flask, render_template_string
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run


@app.route("/")
def hello():
    return "Hello World!"
#         return render_template_string("""
# {% block title %}Home{% endblock %}
# {% block body %}
# <div class="jumbotron">
#   <h1>Flask Is Awesome</h1>
#   <p class="lead">And I'm glad to be learning so much about it!</p>
# </div>
# {% endblock %}
#""")


if __name__ == '__main__':
    app.run()
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style='dark')
import plotly.express as px

df = pd.read_csv("../input/titlel/avocado.csv", encoding='UTF-8', index_col=0)

from pycaret.regression import *
exp_reg = setup(df, target='AveragePrice')

model = create_model('et')
evaluate_model(model)

# compare_models()

# Model 	MAE 	MSE 	RMSE 	R2 	RMSLE 	MAPE 	TT (Sec)
# 0 	Extra Trees Regressor 	0.0757 	0.0126 	0.1120 	0.9226 	0.0449 	0.0558 	4.3985
# 1 	CatBoost Regressor 	0.0863 	0.0139 	0.1180 	0.9142 	0.0469 	0.0628 	4.1328
# 2 	Extreme Gradient Boosting 	0.0904 	0.0155 	0.1243 	0.9049 	0.0497 	0.0661 	5.5836
# 3 	Random Forest 	0.0888 	0.0162 	0.1271 	0.9004 	0.0511 	0.0658 	4.9085
# 4 	Light Gradient Boosting Machine 	0.1008 	0.0186 	0.1362 	0.8858 	0.0547 	0.0743 	0.3815
# 5 	Decision Tree 	0.1229 	0.0349 	0.1867 	0.7852 	0.0749 	0.0902 	0.2130
l_model = load_model('../input/finalma/model')
data = {'Date':[''], 'AveragePrice':[0],'Total Volume':[0], '4046':[0], '4225':[0], '4770':[0], 'Total Bags':[0], 
        'Small Bags':[0], 'Large Bags':[0], 'XLarge Bags':[0], 'type':['organic'], 'year':[''], 'region': ['LasVegas']}
pred = pd.DataFrame(data)

new_prediction = predict_model(l_model, data=pred)
print(new_prediction)
final_model = finalize_model(model);
predict_model(model);
save_model(final_model,'model')
save_model(final_model,'model')