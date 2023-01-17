import numpy as np

import pandas as pd

import plotly.express as px
class MRP(object):

    def __init__(self,demand,Q,lt,pastdue,SRf=False,SRq=None,SRt=None):

        self.demand = demand

        self.period = len(self.demand)

        self.periods = np.arange(self.period)

        self.Q = Q

        self.lt = lt

        self.pastdue = pastdue

        self.SRf = SRf

        self.SRq = SRq

        self.SRt = SRt

        self.GR = self.demand

        self.SR = np.zeros(self.period)

        self.NR= np.zeros(self.period)

        self.POH = np.zeros(self.period)

        self.PORec = np.zeros(self.period)

        self.POR = np.zeros(self.period)

        self.POH[0] =  self.pastdue

        if self.SRf == True:

            self.SR[self.SRt] = self.SRq

    def main(self,name):

        for t in range(1,self.period):

            self.NR[t] = self.GR[t] - self.SR[t] - self.POH[t-1] 

            if self.NR[t]<0:

                self.NR[t] = 0

            self.PORec[t] = self.NR[t]

            self.POH[t] = self.SR[t] + self.PORec[t] + self.POH[t-1] - self.GR[t]

            if self.POH[t] < 0:

                self.POH[t] = 0

            self.POR[t-self.lt] = self.PORec[t]

            if self.POR[t-self.lt] > 0 and self.POR[t-self.lt] < self.Q:

                gecici = self.POR[t-lt]

                self.POR[t-self.lt] = self.Q

                self.POH[t-self.lt+1] = self.Q - gecici

        dict_df = {

            "Periods":self.periods,

            "Gross Reqmts.":self.GR,

            "Sched. Receipts":self.SR,

            "Proj. On Hand":self.POH,

            "Net Reqmts.":self.NR,

            "Planned Receipts":self.PORec,

            "Planned Order Release":self.POR

        }

        df = pd.DataFrame(dict_df)

        

        data = pd.DataFrame()

        data["Periods"] = list(df["Periods"])*6

        dummy= []

        label = np.array(df.columns)

        label = np.delete(label, 0)

        for i in range(len(label)):

            for j in range(len(df.Periods)):

                dummy.append(label[i])

        data["Label"]=pd.Series(dummy)

        value = []

        for i in range(1,len(df.columns)):

            value.append(df.iloc[:,i])

        value = np.array(value).flatten()

        data["Quantitiy"] = value

        

        fig = px.bar(data,x='Periods',y='Quantitiy',color="Label",

                     title="Material Requirements Planning For {}".format(name))

        fig.show()

    

            

        return df.T
KotYelek = MRP(demand=np.array([0,21310,0,10658,34100,0,0,25570]),

             Q=1,

             lt=1,

             pastdue=21310,

             SRf=False,

             SRq=None,

             SRt=None)

KotYelek.main("Kot Yelek")
AnaGovde = MRP(demand=np.array([0,0,0,10658,34100,0,0,25570]),

             Q=1,

             lt=1,

             pastdue=0,

             SRf=False,

             SRq=None,

             SRt=None)

AnaGovde.main("Ana Gövde")
Yaka = MRP(demand=np.array([0,0,0,10658,34100,0,0,25570]),

             Q=2000,

             lt=5,

             pastdue=0,

             SRf=True,

             SRq=49234,

             SRt=2)

Yaka.main("Yaka")
Dugme = MRP(demand=np.array([0,0,0,10658,34100,0,0,25570])*3,

             Q=20000,

             lt=1,

             pastdue=10658,

             SRf=False,

             SRq=None,

             SRt=None)

Dugme.main("Düğme")
Arka = MRP(demand=np.array([0,0,0,10658,34100,0,0,25570]),

             Q=5000,

             lt=2,

             pastdue=0,

             SRf=True,

             SRq=44758,

             SRt=1)

Arka.main("Arka")
On = MRP(demand=np.array([0,0,0,10658,34100,0,0,25570]),

             Q=1000,

             lt=2,

             pastdue=53710,

             SRf=False,

             SRq=None,

             SRt=None)

On.main("Ön")