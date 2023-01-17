import numpy as np 
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import cmath
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt

coron=pd.read_csv('/kaggle/input/wind-coron/coron_all.csv',parse_dates=["time"]).set_index("time")
cortegada=pd.read_csv('/kaggle/input/wind-coron/cortegada_all.csv',parse_dates=["time"]).set_index("time")
join = cortegada.join(coron, lsuffix='_corte', rsuffix='_coron').dropna()
join.columns
#select threshold
threshold=0
#select variable threshold
vars_threshold=['mod_corte','mod_coron',"spd_o_coron","spd_o_corte"]
var_threshold=vars_threshold[1]
mask=join[join[var_threshold]>=threshold]
mae_dir={"dir_o_corte":[mean_absolute_error(mask.dir_o_corte, mask.dir_corte),
                       mean_absolute_error(mask.dir_o_corte, mask.dir_coron)],
        "dir_o_coron":[mean_absolute_error(mask.dir_o_coron, mask.dir_corte),
                      mean_absolute_error(mask.dir_o_coron, mask.dir_coron)]}
pd.DataFrame(mae_dir, index=['dir_corte', 'dir_coron'])
mae_dir={"spd_o_corte":[mean_absolute_error(join.spd_o_corte, join.mod_corte),
                       mean_absolute_error(join.spd_o_corte, join.mod_coron)],
        "spd_o_coron":[mean_absolute_error(join.spd_o_coron, join.mod_corte),
                      mean_absolute_error(join.spd_o_coron, join.mod_coron)]}
pd.DataFrame(mae_dir, index=['spd_corte', 'spd_coron'])
observed_corte=[]
predicted_corte=[]
for intensity, direction in zip(join.spd_o_corte.values,np.radians(join.dir_o_corte.values)):
    observed_corte.append(cmath.rect(intensity,direction))
for intensity, direction in zip(join.mod_corte.values,np.radians(join.dir_corte.values)):
    predicted_corte.append(cmath.rect(intensity,direction))
join["w_corte_o_complex"]=np.asarray(observed_corte)
join["w_corte_p_complex"]= np.asarray(predicted_corte)   
join["w_corte_diff_abs"]=(join["w_corte_o_complex"]-join["w_corte_p_complex"]).abs()
join["w_corte_diff_phase"]=np.angle((join["w_corte_o_complex"]-join["w_corte_p_complex"]),deg=True)

observed_coron=[]
predicted_coron=[]
for intensity, direction in zip(join.spd_o_coron.values,np.radians(join.dir_o_coron.values)):
    observed_coron.append(cmath.rect(intensity,direction))
for intensity, direction in zip(join.mod_coron.values,np.radians(join.dir_coron.values)):
    predicted_coron.append(cmath.rect(intensity,direction))
join["w_coron_o_complex"]=np.asarray(observed_coron)
join["w_coron_p_complex"]= np.asarray(predicted_coron)   
join["w_coron_diff_abs"]=(join["w_coron_o_complex"]-join["w_coron_p_complex"]).abs()
join["w_coron_diff_phase"]=np.angle((join["w_coron_o_complex"]-join["w_coron_p_complex"]),deg=True)
join[["w_corte_diff_abs","w_coron_diff_abs"]].describe()


Y=join["spd_o_coron"]
X=join[["dir_coron","mod_coron"]]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,)
clf = DecisionTreeRegressor(criterion="mae").fit(x_train,y_train) 
y_pred=clf.predict(x_test)
mean_absolute_error(y_test, y_pred)