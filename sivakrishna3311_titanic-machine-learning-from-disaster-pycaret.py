import pandas as pd
import numpy as np
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
!pip install pycaret
from pycaret.classification import *
df=pd.read_csv('../input/titanic/train.csv')
df.head()
len(df[df['Survived']==0])
len(df[df['Survived']==1])
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = df, target = 'Survived')
compare_models()
# creating reidge classifier
ridge= create_model('ridge')
model=tune_model('ridge')
data_unseen = pd.read_csv('../input/titanic/test.csv')
# generate predictions on unseen data
predictions = predict_model(model, data = data_unseen)
predictions
predictions=predictions[['PassengerId','Label']]
predictions.rename(columns={'Label':'Survived'},inplace=True)
predictions
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(predictions)