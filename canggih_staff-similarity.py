import numpy as np
import pandas as pd
def WordVectorSimilarity(ar1,ar2):
    d1 = []    
    for s1 in ar1:
        if s1 != "":
            s1 = s1.lower()
            d1.append(s1)
    d2 = []    
    for s2 in ar2:
        if s2 != "":
            s2 = s2.lower()
            d2.append(s2)
    
    intersect = list(set(d1) & set(d2))
    if len(intersect) == 0:
        count = 1
    else:
        count = len(intersect)
    return count

def ConstructMatrixStaff(data):
    index = data.iloc[:,0]
    staff = data.iloc[:,1]
    dfstaff = pd.DataFrame(0,index=index, columns=index)
    for i in staff.iteritems():
        idata = str(i[1]).split(';')
        for j in staff[0:int(i[0])+1].iteritems():
            jdata = str(j[1]).split(';')
            if i[0] == j[0]:
                dfstaff.iat[int(i[0]),int(j[0])] = -1
            else:
                dfstaff.iat[int(i[0]),int(j[0])] = WordVectorSimilarity(idata,jdata)
    mstaff = np.matrix(dfstaff)
    newstaff = mstaff + np.transpose(mstaff)
    dfnewstaff = pd.DataFrame(newstaff,index=index, columns=index)
    return dfnewstaff
dstaff = pd.read_csv('../input/datastaff-all-share-new.csv',sep='|')
staff_matrix = ConstructMatrixStaff(dstaff)
print(staff_matrix.head(10))
staff_matrix.to_csv('sim_staff.csv')