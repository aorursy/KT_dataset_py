import pandas as pd

coron=pd.read_csv('/kaggle/input/wind-coron/coron_all.csv',parse_dates=["time"]).set_index("time")

!pip install dataprep

from dataprep.eda import plot
plot(coron)
plot(coron,"spd_o",bins=10)
plot(coron,"spd_o","mod",bins=5)