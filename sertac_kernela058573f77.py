import numpy as np # linear algebra

drawYT = lambda x : ''.join([ str(np.random.choice(['Y','T'])) for i in range(x) ])

bir_suru = [drawYT(10) for i in range(10**6)]
ilk_besi_Y = [x for x in bir_suru if x[0:5] == 'YYYYY']
altinci_da_Y = [x for x in ilk_besi_Y if x[5] == 'Y']
altinci_T = [x for x in ilk_besi_Y if x[5] == 'T']
print(f'Toplam {len(bir_suru)} tane observation var.') 

print(f'{len(ilk_besi_Y)} tane ilk besi Y olan observation var.') 

print(f'Bunlarin {len(altinci_da_Y)} tanesinin 6.si da Y. Oran = {len(altinci_da_Y)/len(ilk_besi_Y)}')
print(f'Bunlarin {len(altinci_T)} tanesinin 6.si T. Oran = {len(altinci_T)/len(ilk_besi_Y)}')

T = [0,0,0,0,0,0]
diger_5 = [i[5:10] for i in ilk_besi_Y]

for x in diger_5:
    T[len([x for i in x if i == 'Y'])] += 1
    
print(T)
    
print(f'Bunlarin {T[0]} tanesinin 6.si da Y. Oran = {T[0]/len(ilk_besi_Y)}')
print(f'Bunlarin {T[1]} tanesinin 7.si da Y. Oran = {T[1]/len(ilk_besi_Y)}')
print(f'Bunlarin {T[2]} tanesinin 8.si da Y. Oran = {T[2]/len(ilk_besi_Y)}')
print(f'Bunlarin {T[3]} tanesinin 9.si da Y. Oran = {T[3]/len(ilk_besi_Y)}')
print(f'Bunlarin {T[4]} tanesinin 10.si da Y. Oran = {T[4]/len(ilk_besi_Y)}')
print(f'Bunlarin {T[5]} tanesinin 10.si da Y. Oran = {T[5]/len(ilk_besi_Y)}')

