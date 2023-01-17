from pulp import*
import pandas as pd

prod = ['HiFi-1','HiFi-2']
ws = [1,2,3]

prodws = []
for i in prod:
    for j in ws:
        prodws.append([i,j])

tottime = 480
prodwstime = [['HiFi-1',1,6],['HiFi-1',2,5],['HiFi-1',3,4],['HiFi-2',1,4],['HiFi-2',2,5],['HiFi-2',3,6]]
maint = [[1,0.10],[2,0.14],[3,0.12]]

prodwstimedf = pd.DataFrame(prodwstime,columns=['Product','WorkStation','PT'])
prodwstimedf = prodwstimedf.set_index(['Product','WorkStation'])

maintdf = pd.DataFrame(maint,columns=['WorkStation','MaintProp'])
maintdf = maintdf.set_index(['WorkStation'])

class ProdMixMinIdleTime:
    
    def __init__(self,prod,ws,prodws,tottime,prodwstimedf,maintdf):
        
        self.prod = prod
        
        self.ws = ws
        
        self.prodws = prodws
        
        self.tottime = tottime
        
        self.prodwstimedf = prodwstimedf
        
        self.maintdf = maintdf
        
        self.prob = LpProblem("Minimize Idle Time",LpMinimize)
        
        self.x = LpVariable.dicts("Product_units_variable", ((i) for i in self.prod),lowBound=0,cat='Integer')
        
        self.s = LpVariable.dicts("Workstation_slack_variable", ((i) for i in self.ws),lowBound=0,cat='Continuous')
               
        self.prob += lpSum([self.s[i] for i in self.ws])
        
        for i in self.ws:
            self.prob += lpSum([self.x[j] * self.prodwstimedf.loc[(j,a),'PT'] for j,a in self.prodws if a == i]) + self.s[i] == self.tottime * (1 - self.maintdf.loc[(i),'MaintProp'])
        
        
    def solve(self):
        self.prob.solve()
        self.prob.writeLP("ProdMixMinIdleTime.lp")
        
    def status(self):
        return LpStatus[self.prob.status]
        
    def objective(self):
        return value(self.prob.objective)
        
    def returnVar(self):
        var = []
        for i,v in enumerate(self.prob.variables()):
            var.append([v.name,v.varValue])
        return var

mod = ProdMixMinIdleTime(prod,ws,prodws,tottime,prodwstimedf,maintdf)
mod.solve()
status = mod.status()
variables = mod.returnVar()
objective = mod.objective()