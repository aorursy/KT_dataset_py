# Import Libraries

import pandas as pd

import pickle

from category_encoders import *

import matplotlib.pyplot as plt

# %matplotlib inline

import seaborn as sns
train_df=pd.read_csv("../input/50-startups/50_Startups.csv")
train_df.head(4)

enc = OrdinalEncoder().fit(train_df)
df_train_encoded = enc.transform(train_df)



df_train_encoded.head(5)
corremat = df_train_encoded.corr()

top_corr_features = corremat.index[(corremat['Profit'])> .1]

plt.figure(figsize=(10,10))

g= sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap='viridis',linewidths=.5)
X=df_train_encoded.iloc[:, :-1].values
y = df_train_encoded.iloc[:, 4].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
df =pd.DataFrame(data=y_test,columns=['y_test'])

df['y_pred'] = y_pred

df
regressor.predict([[165349.20,	136897.80,	471784.10,1]])
import pickle

pickle.dump(regressor,open('startup_prediction.pkl','wb'))

"""

<!DOCTYPE html>

<html >

<!--From https://codepen.io/frytyler/pen/EGdtg-->

<head>

  <meta charset="UTF-8">

  <title>ML API</title>

  <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>

<link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>

<link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>

<link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>

  

</head>



<style>

body {background-color: Lavender ;}

h1   {color: DarkBlue;}

h3   {color: FireBrick   ;}

p    {color: red;}

</style>

<body>

 <div class="login">

	<h1>Predict Profit for Startup</h1>



     <!-- Main Input For Receiving Query to our ML -->

    <form action="{{ url_for('predict')}}"method="post">

	

	<h3>Enter 1 to select the State  </h3>

    	

		

	

		<br>

		<br>

		<input type="text" name="R&D Spent" placeholder="R&D Expence" required="required" />

        <input type="text" name="Administration Expence" placeholder="Administration Expence" required="required" />

		<input type="text" name="Marketing Expence" placeholder="Marketing Expence" required="required" />	

		

		

		<label for="relation">State</label>

<select id="relation" name="relation">

	<option value=1>New York</option>

	<option value=2>California</option>

	<option value=3>Florida</option>

	</select>

		



		

        <button type="submit" required="required" class="btn btn-primary btn-block btn-large">Predict </button>

    </form>



   <br>

   <br>

   {{ prediction_text }}



 </div>





</body>

</html>

"""


"""



import numpy as np

from flask import Flask,request,jsonify,render_template

import pickle



app = Flask(__name__)

salary_deploy = pickle.load(open('startup_prediction.pkl','rb'))



@app.route('/')

def home():

    return render_template('index.html')



@app.route('/predict',methods = ['POST'])



def predict():

    int_features = [float(x) for x in request.form.values()]

    final_features = [np.array(int_features)]

    prediction = salary_deploy.predict(final_features)

    

    output = round(prediction[0],2)

    

    return render_template('index.html',prediction_text ='Profit for the startup should be $ {}'.format(output))





@app.route('/predict_api',methods=['POST'])

def predict_api():



    data = request.get_json(force=True)

    prediction = salary_deploy.predict([np.array(list(data.values()))])



    output = prediction[0]

    return jsonify(output)





if __name__ == '__main__':

    app.run(debug=True)

    """
#Hardcoded json input for the direct api call



"""

import requests



url = 'http://localhost:5000//predict_api'

r = requests.post(url,json={'R&D Spend':122, 'Administration':333, 'Marketing Spend':444,'State':1})



print(r.json())

"""
