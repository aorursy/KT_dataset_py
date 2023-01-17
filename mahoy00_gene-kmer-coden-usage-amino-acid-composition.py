a=['ATATCGTTATTT','TATACATTG','AGAGAGCTC']
lister=['AA','TA','GA','CA','AT','TT','GT','CT','AG','TG','GG','CG','AC','TC','GC','CC']
dict={}
for i in lister:
    dict[i]=0
print(dict)
double_list=[]
for i in a:
    new_list=[]
    count=0
    for j in range(len(i)-2+1):
        dict[i[j:j+2]]+=1
        count+=1
    
    print(dict)
    print(count)
    for k in dict.keys():
        dict[k]=round(dict[k]/count,2)
    print(dict)
    for v in dict.values():
        new_list.append(v)
    print(new_list)
    double_list.append(new_list)
    print(double_list)
    for k in dict.keys():
        dict[k]=0
print(double_list)
import pandas as pd
df = pd.DataFrame(double_list) 
print(df)
lister=['A','T','G','C']
dict={}
for i in lister:
    dict[i]=0
print(dict)
one_list=[]
for i in a:
    new_list=[]
    count=0
    for j in range(len(i)):
        dict[i[j]]+=1
        count+=1
    
    print(dict)
    print(count)
    for k in dict.keys():
        dict[k]=round(dict[k]/count,2)
    print(dict)
    for v in dict.values():
        new_list.append(v)
    print(new_list)
    one_list.append(new_list)
    print(one_list)
    for k in dict.keys():
        dict[k]=0
print(one_list)
df1 = pd.DataFrame(one_list) 
print(df1)
lister=['AAA','TAA','GAA','CAA','ATA','TTA','GTA','CTA','AGA','TGA','GGA','CGA','ACA','TCA','GCA','CCA','AAT','TAT','GAT','CAT','ATT','TTT','GTT','CTT','AGT','TGT','GGT','CGT','ACT','TCT','GCT','CCT','AAG','TAG','GAG','CAG','ATG','TTG','GTG','CTG','AGG','TGG','GGG','CGG','ACG','TCG','GCG','CCG','AAC','TAC','GAC','CAC','ATC','TTC','GTC','CTC','AGC','TGC','GGC','CGC','ACC','TCC','GCC','CCC']


dict={}
for i in lister:
    dict[i]=0
print(dict)
triple_list=[]
for i in a:
    new_list=[]
    count=0
    for j in range(len(i)-3+1):
        dict[i[j:j+3]]+=1
        count+=1
    
    print(dict)
    print(count)
    for k in dict.keys():
        dict[k]=round(dict[k]/count,2)
    print(dict)
    for v in dict.values():
        new_list.append(v)
    print(new_list)
    triple_list.append(new_list)
    print(triple_list)
    for k in dict.keys():
        dict[k]=0
print(triple_list)
df3 = pd.DataFrame(triple_list) 
print(df3)

frames=[df1,df,df3]
result=pd.concat(frames,axis=1)
print(result)
lister=result.values.tolist()
print(lister)
lister=['AAA','TAA','GAA','CAA','ATA','TTA','GTA','CTA','AGA','TGA','GGA','CGA','ACA','TCA','GCA','CCA','AAT','TAT','GAT','CAT','ATT','TTT','GTT','CTT','AGT','TGT','GGT','CGT','ACT','TCT','GCT','CCT','AAG','TAG','GAG','CAG','ATG','TTG','GTG','CTG','AGG','TGG','GGG','CGG','ACG','TCG','GCG','CCG','AAC','TAC','GAC','CAC','ATC','TTC','GTC','CTC','AGC','TGC','GGC','CGC','ACC','TCC','GCC','CCC']
dict={}
for i in lister:
    dict[i]=0
print(dict)

triple_list=[]
for i in a:
    new_list=[]
    count=0
    for j in range(0,len(i)-3+1,3):
        dict[i[j:j+3]]+=1
        count+=1
    
    print(dict)
    print(count)
    for k in dict.keys():
        dict[k]=round(dict[k]/count,2)
    print(dict)
    for v in dict.values():
        new_list.append(v)
    print(new_list)
    triple_list.append(new_list)
    print(triple_list)
    for k in dict.keys():
        dict[k]=0

print(triple_list)
df4 = pd.DataFrame(triple_list) 
print(df4)
a=['ATATCGTTATTT','TATACATTG','AGAGAGCTC']

lister=[]
for i in a:
    seq=''
    for j in range(0,len(i)-2,3):
        if (i[j]=='T' and i[j+1]=='T' and i[j+2]=='T') or(i[j]=='T' and i[j+1]=='T' and i[j+2]=='C') :
            seq=seq+'F'
        elif(i[j]=='T' and i[j+1]=='T' and i[j+2]=='A')or(i[j]=='T' and i[j+1]=='T' and i[j+2]=='G')or(i[j]=='C' and i[j+1]=='T'):
            seq=seq+'L'
        elif(i[j]=='A' and i[j+1]=='T' and i[j+2]=='T')or(i[j]=='A' and i[j+1]=='T' and i[j+2]=='C')or(i[j]=='A' and i[j+1]=='T' and i[j+2]=='A'):
            seq=seq+'I'
        elif(i[j]=='A' and i[j+1]=='T' and i[j+2]=='G'):
            seq=seq+'M'
        elif(i[j]=='G' and i[j+1]=='T'):
            seq=seq+'V'
        elif(i[j]=='T' and i[j+1]=='C')or(i[j]=='A' and i[j+1]=='G' and i[j+2]=='T')or(i[j]=='A' and i[j+1]=='G' and i[j+2]=='C'):
            seq=seq+'S'
        elif(i[j]=='C' and i[j+1]=='C'):
            seq=seq+'P'
        elif(i[j]=='A' and i[j+1]=='C'):
            seq=seq+'T'
        elif(i[j]=='G' and i[j+1]=='C'):
            seq=seq+'A'
        elif(i[j]=='T' and i[j+1]=='A' and i[j+2]=='T')or(i[j]=='T' and i[j+1]=='A' and i[j+2]=='C'):
            seq=seq+'Y'
        elif(i[j]=='C' and i[j+1]=='A' and i[j+2]=='T')or(i[j]=='C' and i[j+1]=='A' and i[j+2]=='C'):
            seq=seq+'H'
        elif(i[j]=='C' and i[j+1]=='A' and i[j+2]=='A')or(i[j]=='C' and i[j+1]=='A' and i[j+2]=='G'):
            seq=seq+'Q'
        elif(i[j]=='A' and i[j+1]=='A' and i[j+2]=='T')or(i[j]=='A' and i[j+1]=='A' and i[j+2]=='C'):
            seq=seq+'N'
        elif(i[j]=='A' and i[j+1]=='A' and i[j+2]=='A')or(i[j]=='A' and i[j+1]=='A' and i[j+2]=='G'):
            seq=seq+'K'
        elif(i[j]=='G' and i[j+1]=='A' and i[j+2]=='T')or(i[j]=='G' and i[j+1]=='A' and i[j+2]=='C'):
            seq=seq+'D'
        elif(i[j]=='G' and i[j+1]=='A' and i[j+2]=='A')or(i[j]=='G' and i[j+1]=='A' and i[j+2]=='G'):
            seq=seq+'E'
        elif(i[j]=='T' and i[j+1]=='G' and i[j+2]=='T')or(i[j]=='T' and i[j+1]=='G' and i[j+2]=='C'):
            seq=seq+'C'
        elif(i[j]=='T' and i[j+1]=='G' and i[j+2]=='G'):
            seq=seq+'W'
        elif(i[j]=='C' and i[j+1]=='G')or(i[j]=='A' and i[j+1]=='G' and i[j+2]=='A')or(i[j]=='A' and i[j+1]=='G' and i[j+2]=='G'):
            seq=seq+'R'
        elif(i[j]=='G' and i[j+1]=='G'):
            seq=seq+'G'
        
    lister.append(seq)

print(lister)
aa_list=['F','L','I','M','V','S','P','T','A','Y','H','Q','N','K','D','E','C','W','R','G']

dict={}
for i in aa_list:
    dict[i]=0
print(dict)

one_list=[]
for i in lister:
    new_list=[]
    count=0
    for j in range(len(i)):
        dict[i[j]]+=1
        count+=1
    
    print(dict)
    print(count)
    for k in dict.keys():
        dict[k]=round(dict[k]/count,2)
    print(dict)
    for v in dict.values():
        new_list.append(v)
    print(new_list)
    one_list.append(new_list)
    print(one_list)
    for k in dict.keys():
        dict[k]=0

df5=pd.DataFrame(one_list) 
print(df5)