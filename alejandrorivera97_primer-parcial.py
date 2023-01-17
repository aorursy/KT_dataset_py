



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import math

#Lista de poblaciones por estado 

p_aguas=[719659,862720,944285,1065416,1184996]

p_bajacal=[719659,862720,944285,1065416,1184996]

p_bajacal_s=[317764,375494,424041,512170,637026]

p_campeche=[535185,642516,690689,754730,822441]

p_coahila=[1972340,2173775,2298070,2495200,2748391]

p_colima=[428510,488028,542627,567996,650555]

p_chiapas=[3210496,3584786,3920892,4293459,4796580]

p_chihuahua=[2441873,2793537,3052907,3241444,3406465]

p_cdmx=[8235744,8489007,8605239,8720916,8851080]

p_durango=[1349378,1431748,1448661,1509117,1632934]

p_guanajuato=[3982593,4406568,4663032,4893812,5486372]

p_guerrero=[2620637,2916567,3079649,3115202,3388768]

p_hidalgo=[1888366,2112473,2235591,2345514,2665018]

p_jalisco=[5302689,5991176,6322002,6752113,7350682]

p_e_mex=[9815795,11707964,13096686,14007495,15175862]

p_michoacan=[3548199,3870604,3985667,3966073,4351037]

p_morelos=[1195059,1442662,1555296,1612899,1777227]

p_nayarit=[824643,896702,920185,949684,1084979]

p_n_leon=[3098736,3550114,3834141,4199292,4653458]

p_oaxaca=[3019560,3228895,3438765,3506821,3801962]

p_puebla=[4126101,4624365,5076686,5383133,5779829]

p_queretaro=[1051235,1250476,1404306,1598139,1827937]

p_q_roo=[493277,703536,874963,1135309,1325578]

p_san_luis=[2003187,2200763,2299360,2410414,2585518]

p_sinaloa=[2204054,2425675,2536844,2608442,2767761]

p_sonora=[1823606,2085536,2216969,2394861,2662480]

p_tabasco=[1501744,1748769,1891829,1989969,2238603]

p_tamaulipas=[2249581,2527328,2753222,3024238,3268554]

p_tlaxcala=[761277,883924,962646,1068207,1169936]

p_veracruz=[6228239,6737324,6908975,7110214,7643194]

p_yucatan=[1362940,1556622,1658210,1818948,1955577]

p_zacatecas=[1276323,1336496,1353610,1367692,1490668]

#Vamos a hacer una lista con estas litas para poder aplicar una función y sacar la tasa de crecimiento poblacional promedio. 

estados=["Aguascalientes", "Baja California","Baja California Sur","Campeche","Coahila","Colima","Chiapas","Chihuahua",

        "CDMX","Durango","Guanajuato","Guerrero","Hidalgo","Jalisco","Estado de Mexico","Michoacan","Morelos","Nayarit",

        "Nuevo León","Oaxaca","Puebla","Queretaro","Quintana Roo","San Luis Potosi","Sinaloa","Sonora","Tabasco","Tamaulipas",

        "Tlaxcala","Veracruz","Yucatan","Zacatecas"]

p_mexico=[p_aguas,p_bajacal,p_bajacal_s,p_campeche,p_coahila,p_colima,p_chiapas,p_chihuahua,p_cdmx,p_durango,p_guanajuato,

         p_guerrero,p_hidalgo,p_jalisco,p_e_mex,p_michoacan,p_morelos,p_nayarit,p_n_leon,p_oaxaca,p_puebla,p_queretaro,p_q_roo,p_san_luis,

         p_sinaloa,p_sonora,p_tabasco,p_tamaulipas,p_tlaxcala,p_veracruz,p_yucatan,p_zacatecas]

#Vamos a crear una función para poder calcular la tasa de crecimiento promedio anual por estado y tomarla como el valor n para el modelo

def tasa(lista):

    vacia=[]

    for i in range(5):

        if i<4:

            anterior=lista[i]

            despues=lista[i+1]

            n_anual=(despues/anterior)-1

            vacia.append(n_anual)

    contar=0

    for con in vacia: #tenemos que convertir la tasa quinquinal a efectiva anual

        con=((con+1)**(1/5))-1

        vacia[contar]=con

        contar+=1

        

    r_anual=((vacia[0]+1)*(vacia[1]+1)*(vacia[2]+1)*(vacia[3]+1))**(1/5) #sacamos el promedio geométrico para aproximar la n

    return (r_anual-1)

        
#Ahora vamos a crear una lista con todas las n por estado

n_estados=[]

for estado in p_mexico:

    n=round(tasa(estado),4)

    n_estados.append(n)

print(n_estados)
ultima_pob=[]

for estado in p_mexico:

    ultima=estado[4]

    ultima_pob.append(ultima)

    

ultima2005_pob=[]

for estado in p_mexico:

    ultima=estado[3]

    ultima2005_pob.append(ultima)

prediccion_pob=pd.DataFrame({"n":n_estados,"2005":ultima2005_pob, "2010":ultima_pob},index=estados)



for i in range(4):

    prediccion_pob[str(2005+1+i)]=(prediccion_pob["n"]+1)*prediccion_pob[str(2005+i)]

    

for i in range(8):

    prediccion_pob[str(2010+1+i)]=(prediccion_pob["n"]+1)*prediccion_pob[str(2010+i)]
#Estos ya están evaluados por el INEGI, solo que están de mayor a menor y los tengo que poner en el orden de los estados

escolaridad=[9.7, 9.8, 9.9, 9.1, 9.9, 9.5, 7.3, 9.5, 11.1, 9.1, 8.4, 7.8,

            8.7, 9.2, 9.5, 7.9, 9.3, 9.2, 10.3, 7.5, 8.5, 9.6, 9.6, 8.8, 

            9.6, 10.0, 9.3, 9.5, 9.3,8.2,8.8, 8.6]

print(len(escolaridad))
h=[]

for year in escolaridad:

    cali=round(math.exp(.1*year),4)

    h.append(cali)

print(h)
pib_aguas=[121197.634,126554.147,129628.028,138111.823,150305.117,150949.983,143253.947,152205.212,158934.494,167705.967,172820.491,191038.591,198175.395,216221.106,224991.631,232547.255]

pib_bajacal=[399514.624,423005.499,433008.166,456019.296,461581.274,457556.565,407745.948,428162.546,440700.662,456024.472,465524.695,473362.348,505937.657,528019.778,545081.869,557853.091]

pib_bajacal_s=[76047.593,81546.009,87397.67,93655.622,106199.426,108975.356,108338.811,110656.4,114707.762,117345.833,115027.644,114871.335,130096.58,134227.782,148655.847,174246.239]

pib_campeche=[1047511.322,1059561.025,1038533.775,1014280.35,947575.493,867231.044,780757.429,753968.591,726503.856,714787.065,721085.063,687268.582,638740.803,601066.487,537722.196,528896.04]

pib_coahila=[436573.518,449143.882,458867.907,480488.194,500478.657,498326.775,421327.489,489951.776,523207.112,549551.802,538206.987,565824.825,573850.068,582858.427,613737.775,621735.14]

pib_colima=[67732.93,67794.051,68258.171,72533.785,77526.37,78953.717,76446.9,81992.178,87944.969,90540.289,91422.446,93707.678,95357.745,101200.627,105464.29,108110.288]

pib_chiapas=[248123.227,238375.89,240279.559,248414.302,252536.304,258289.89,256698.362,270989.331,279446.584,284733.625,280925.273,295158.113,290463.614,290645.764,282833.403,276850.552]

pib_chihuahua=[360426.663,376662.637,389210.794,419631.539,434649.92,440792.887,401079.179,417796.42,427430.027,459166.22,476290.197,486857.759,515187.553,540446.937,558439.596,567395.31]

pib_cdmx=[2132929.372,2226949.736,2258091.583,2374722.886,2408565.865,2450391.202,2362516.439,2446910.439,2533806.893,2633934.642,2673066.331,2729859.454,2836540.252,2961597.62,3046955.925,3129179.877]

pib_durango=[152922.727,157662.281,155001.886,160388.081,162709.802,165722.989,163083.647,169268.084,176314.713,182943.056,189052.812,193539.476,194989.457,202282.368,199507.56,201196.05]

pib_guanajuato=[438354.387,450953.155,454625.559,477646.929,488729.604,503024.423,481674.906,517168.681,548163.17,570921.987,594575.532,621005.836,661221.488,689277.314,721583.582,729919.389]

pib_guerrero=[182713.981,192557.837,195219.889,199540.65,204879.83,208284.821,201239.316,211890.535,214478.174,218118.481,218811.378,229021.256,232024.32,236941.183,235931.007,242952.942]

pib_hidalgo=[179553.378,191549.579,190073.779,195404.535,201655.418,208800.335,195581.126,206303.584,214569.188,222797.005,230982.767,240079.594,253581.601,264155.704,264114.693,272561.297]

pib_jalisco=[794957.322,819238.31,842128.813,886009.726,913139.834,918573.456,870319.1,925371.837,953148.056,995285.999,1018578.607,1062083.776,1107681.987,1162001.062,1192385.651,1226570.141]

pib_e_mex=[ 1048403.59,1073840.81,1099376.794,1150701.95,1184658.428,1198144.35,1138727.918,1226813.687,1283448.197,1339994.611,1365154.229,1405514.291,1438521.879,1481449.968,1542591.568,1584063.785 ]

pib_michoacan=[294468.306,301021.642,306026.395,320451.426,328272.19,334657.914,317003.042,329767.26,343275.664,352030.387,359465.987,383195.32,391667.431,408268.918,422124.7,430351.926]

pib_morelos=[158055.834,159734.875,171279.471,168177.81,169325.362,169672.507,168348.348,174984.467,174678.088,175717.837,182126.143,184150.263,186472.282,192344.911,202253.571,201299.745]

pib_nayarit=[76105.196,86879.422,90269.847,92165.697,91675.276,98292.89,93038.497,97786.134,100704.16,100800.218,103627.459,109267.967,114883.654,119686.74,121602.292,120415.547]

pib_n_leon=[803888.528,849841.554,885438.787,946837.916,1004636.521,1020366.768,952725.778,1025184.258,1069812.268,1113817.766,1124999.893,1162064.865,1219286.846,1239320.882,1278690.616,1324742.965]

pib_oaxaca=[202963.936,210757.192,213677.217,218023.575,219815.363,226633.787,224510.705,228089.144,234955.838,239680.171,245515.976,250555.694,260507.541,256411.476,250441.891,262170.238]

pib_puebla=[395907.258,405907.943,428179.294,447201.403,465818.722,468969.471,432578.676,469967.84,493353.224,524226.058,519256.535,524307.546,539447.247,553071.57,587299.172,601167.726]

pib_queretaro=[212106.713,227917.275,243311.19,258448.271,271622.051,278348.405,270311.398,287403.169,308865.228,318294.372,319989.728,345653.114,369835.745,385705.111,401850.471,413808.118]

pib_q_roo=[144233.02,156941.599,163681.664,174364.522,192904.582,203018.581,185671.746,195148.828,206053.849,215709.871,225272.667,233661.538,245512.275,263378.223,274581.265,288571.661]

pib_san_luis=[224280.267,237464.465,247240.388,258649.446,264315.631,270024.005,255845.961,269397.22,283881.746,297293.962,307896.47,315395.653,330163.06,342485.817,358909.59,374094.011]

pib_sinaloa=[268247.066,284658.629,285708.482,294951.695,305622.537,316380.798,303066.175,312655.113,318762.638,330191.387,334097.307,341211.831,361904.436,381837.764,385283.092,395849.293]

pib_sonora=[365533.727,384484.325,404881.254,429625.189,435397.343,436717.189,410374.271,431501.919,471510.171,495926.031,510315.674,519083.264,537497.667,567563.28,572012.567,578668.79]

pib_tabasco=[374891.404,391243.029,421079.732,445309.616,454079.113,475202.827,495944.159,525011.917,549751.131,564003.811,553628.205,562825.471,559067.971,529902.799,505141.849,463733.208]

pib_tamaulipas=[391574.439,410112.746,436490.981,449609.24,464007.812,483350.662,439739.32,448215.116,456768.535,466371.322,473241.401,478550.581,490612.616,490716.237,490025.603,500964.947]

pib_tlaxcala=[83254.186,89789.73,79279.279,77889.934,79019.573,83246.79,81739.53,88809.886,86031.739,89918.601,87657.644,90496.304,96609.075,97224.235,96251.254,99537.521]

pib_veracruz=[613590.201,643859.491,648906.398,684557.368,705608.578,704314.03,688981.395,718148.55,746817.731,779730.434,781357.276,790857.58,803983.3,803426.369,797002.138,815080.511]

pib_yucatan=[161636.497,169013.747,176907.681,185462.602,191217.075,193158.623,189365.458,196149.981,202893.85,214700.599,215788.237,223091.269,232221.157,242413.381,250889.207,258936.101]

pib_zacatecas=[101406.31,105664.685,105661.441,112108.368,115935.139,126383.577,130512,144730.56,144876.886,148728.64,146858.788,157068.407,159227.206,156595.23,156172.432,155967.054]



pib_mexico=[pib_aguas,pib_bajacal,pib_bajacal_s,pib_campeche,pib_coahila,pib_colima,pib_chiapas,pib_chihuahua,pib_cdmx,pib_durango,pib_guanajuato,

         pib_guerrero,pib_hidalgo,pib_jalisco,pib_e_mex,pib_michoacan,pib_morelos,pib_nayarit,pib_n_leon,pib_oaxaca,pib_puebla,pib_queretaro,pib_q_roo,pib_san_luis,

         pib_sinaloa,pib_sonora,pib_tabasco,pib_tamaulipas,pib_tlaxcala,pib_veracruz,pib_yucatan,pib_zacatecas]

pib_2005=[]

for estado in pib_mexico:

    pib=estado[2]

    pib_2005.append(pib)

    

pib_2006=[]

for estado in pib_mexico:

    pib=estado[3]

    pib_2006.append(pib)

    

pib_2007=[]

for estado in pib_mexico:

    pib=estado[4]

    pib_2007.append(pib)

    

pib_2008=[]

for estado in pib_mexico:

    pib=estado[5]

    pib_2008.append(pib)

    

pib_2009=[]

for estado in pib_mexico:

    pib=estado[6]

    pib_2009.append(pib)

    

    

pib_2010=[]

for estado in pib_mexico:

    pib=estado[7]

    pib_2010.append(pib)

    

pib_2011=[]

for estado in pib_mexico:

    pib=estado[8]

    pib_2011.append(pib)

    

pib_2012=[]

for estado in pib_mexico:

    pib=estado[9]

    pib_2012.append(pib)

    

pib_2013=[]

for estado in pib_mexico:

    pib=estado[10]

    pib_2013.append(pib)

    

pib_2014=[]

for estado in pib_mexico:

    pib=estado[11]

    pib_2014.append(pib)

    

pib_2015=[]

for estado in pib_mexico:

    pib=estado[12]

    pib_2015.append(pib)

    

    

pib_2016=[]

for estado in pib_mexico:

    pib=estado[13]

    pib_2016.append(pib)

    

pib_2017=[]

for estado in pib_mexico:

    pib=estado[14]

    pib_2017.append(pib)

    

pib_2018=[]

for estado in pib_mexico:

    pib=estado[15]

    pib_2018.append(pib)

        
pib_anual=pd.DataFrame({"2005":pib_2005, "2006":pib_2006, "2007":pib_2007, "2008":pib_2008, "2009":pib_2009

                       , "2010":pib_2010, "2011":pib_2011, "2012":pib_2012, "2013":pib_2013, "2014":pib_2014,

                        "2015":pib_2015, "2016":pib_2016, "2017":pib_2017, "2018":pib_2018},index=estados)

pib_per_ca=pd.DataFrame(index=estados)

for i in range(14):

    pib_per_ca[str(2005+i)]=(pib_anual[str(2005+i)]*1000000)/prediccion_pob[str(2005+i)]

pib_per_ca.describe()

g_anuales=pd.DataFrame(index=estados)

for i in range(13):

    if i<13:

        g_anuales[str(i)]=pib_per_ca[str(2005+i+1)]/pib_per_ca[str(2005+i)]

g_anuales["g_estimada"]=((g_anuales["0"]*g_anuales["1"]*g_anuales["2"]*g_anuales["3"]*g_anuales["4"]*g_anuales["5"]*g_anuales["6"]*g_anuales["7"]*g_anuales["8"]*g_anuales["9"]*g_anuales["10"]*g_anuales["11"]*g_anuales["12"])**(1/13))-1

g_anuales["g_estimada"].head()

#Las cifras están expresadas en millones de pesos.

capital_total=[

       458300.571,

1389716.057,

351857.49,

1625187.775,

1329858.598,

279254.41,

1146822.956,

1409795.286,

8473474.931,

520866.602,

1536867.562,

720369.006,

848377.642,

2994190.22,

5122659.601,

946359.086,

666252.746,

451075.745,

3213957.902,

1316348.023,

1554609.768,

891765.302,

543586.65,

895689.178,

933307.084,

1470603.096,

2148326.173,

1635238.459,

320563.643,

3244941.299,

592106.489,

379067.83

]
s_promedio=[0.120186818,

0.098245974,

0.122404664,

0.076781081,

0.109700957,

0.294654955,

0.127836641,

0.081046112,

0.06408819,

0.106191482,

0.067383243,

0.208731392,

0.091999581,

0.066900378,

0.07014651,

0.064351454,

0.084520102,

0.141995523,

0.083868368,

0.078563853,

0.113736983,

0.092285029,

0.103044459,

0.09615553,

0.094102565,

0.082230828,

0.078328131,

0.101259126,

0.073053272,

0.146397354,

0.08403498,

0.15740171]
analisis=pd.DataFrame({"n":n_estados, "u":escolaridad,"h":h,"K_total":capital_total, "s":s_promedio},index=estados)



exponente=(2/3)/(1-(2/3))



analisis["x"]=analisis["n"]+.075



analisis["sgorro"]=analisis["s"]/analisis.loc["CDMX","s"]



analisis["xgorro"]=analisis["x"]/analisis.loc["CDMX","x"]



analisis["hgorro"]=analisis["h"]/analisis.loc["CDMX","h"]



analisis["g_estimada"]=g_anuales["g_estimada"]



analisis["y"]=pib_per_ca["2013"]



analisis["k2013"]=(analisis["K_total"]*1000000)/prediccion_pob["2013"]



analisis["A"]=((analisis["y"]/analisis["k2013"])**exponente)*(analisis["y"]/analisis["h"])



analisis["lnA"]=np.log(analisis["A"])



analisis["lny_real"]=np.log(analisis["y"])



dentro=analisis["s"]/analisis["x"]



dentro=dentro**exponente



analisis["lny_estimada"]=np.log(dentro*analisis["h"])



analisis["lny_real_relativa"]=analisis["lny_real"]/analisis.loc["CDMX","lny_real"]



analisis["lny_estimada_relativa"]=analisis["lny_estimada"]/analisis.loc["CDMX","lny_estimada"]



analisis["lnA_relativa"]=analisis["lnA"]/analisis.loc["CDMX","lnA"]



analisis.head()
estimacion=analisis.loc[:,["sgorro","xgorro","hgorro","y","A","g_estimada"]]
estimacion["A_rel"]=estimacion["A"]/estimacion.loc["CDMX","A"]



estimacion["lny"]=np.log(estimacion["y"])



estimacion["y_rel_real"]=estimacion["y"]/estimacion.loc["CDMX","y"]



paren=(estimacion["sgorro"]/analisis["xgorro"]) 

parentesis=paren**exponente



estimacion["y_estimada"]=parentesis*estimacion["hgorro"]



estimacion["y_estimada_A"]=parentesis*estimacion["hgorro"]*estimacion["A_rel"]







print(estimacion.loc[:,["y_estimada","y_rel_real","y_estimada_A","A_rel"]])
fig,ax=plt.subplots()

ax.scatter(analisis["lny_real_relativa"],analisis["lny_estimada_relativa"])

ax.set_xlabel("Producto per cápita Relativo de los Estados ")

ax.set_ylabel('Estimación del valor relativo Y/L')

plt.title('Caracterización del modelo neoclásico a los estados de México')

plt.show()



fig,productividad=plt.subplots()

productividad.scatter(analisis["lny_real_relativa"],analisis["lnA_relativa"])

productividad.set_xlabel("Y/L Relativo con logaritmo")

productividad.set_ylabel("A relativa estimada con logaritmo")

plt.title('Niveles de productividad')

plt.show()
fig,escolaridad=plt.subplots()

escolaridad.scatter(analisis["y"],analisis["u"])

escolaridad.set_xlabel("PIB per cápita 2013")

escolaridad.set_ylabel("Años promedio de estudio")

plt.title('Relación PIB per cápita con escolaridad')

plt.show()
analisis_pib_1=pib_per_ca.iloc[[3,8,4,18,25,11,14,28,19,6],:]

print(analisis_pib_1)

analisis_pib=analisis_pib_1.T

print(analisis_pib)

analisis_convergencia=analisis.iloc[[8,3,4,18,25,11,14,28,19,6],:]

analisis_convergencia.head()

analisis_convergencia_altos=analisis.iloc[[8,4,18,25],:]

analisis_convergencia_altos.head()

analisis_convergencia_bajos=analisis.iloc[[11,14,28,19,6],:]
fig_1=plt.figure(figsize=(8,8))

tiempo=pd.DataFrame({"Años":[2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]})

plt.plot(tiempo, analisis_pib["CDMX"],label="CDMX")

plt.plot(tiempo, analisis_pib["Campeche"],label="Campeche")

plt.plot(tiempo, analisis_pib["Coahila"],label="Coahila")

plt.plot(tiempo, analisis_pib["Nuevo León"],label="Nuevo León")

plt.plot(tiempo, analisis_pib["Sonora"],label="Sonora")

plt.legend(loc="upper right")

plt.xlabel('Año')

plt.ylabel('PIB per cápita')

plt.title('Comportamiento del PIB per cápita')

plt.show()
fig_1=plt.figure(figsize=(8,8))

colors=["red","blue","green","orange","black"]

tiempo=pd.DataFrame({"Años":[2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]})

plt.plot(tiempo, analisis_pib["CDMX"],label="CDMX")

plt.plot(tiempo, analisis_pib["Coahila"],label="Coahila")

plt.plot(tiempo, analisis_pib["Nuevo León"],label="Nuevo León")

plt.plot(tiempo, analisis_pib["Sonora"],label="Sonora")

plt.legend(loc="upper left")

plt.xlabel('Año')

plt.ylabel('PIB per cápita')

plt.title('Comportamiento del PIB per cápita de los Estados con mayor PIB')

plt.show()
fig,convergencia_altos=plt.subplots()

convergencia_altos.scatter(np.log(analisis_convergencia_altos["y"]),analisis_convergencia_altos["g_estimada"])

convergencia_altos.set_xlabel("PIB per cápita inicial 2005")

convergencia_altos.set_ylabel("Promedio de g 2005-2018")

plt.title('Convergencia de los más altos')

plt.show()
fig_1=plt.figure(figsize=(8,8))

colors=["red","blue","green","orange","black"]

tiempo=pd.DataFrame({"Años":[2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]})

plt.plot(tiempo, analisis_pib["Estado de Mexico"],label="Estado de México")

plt.plot(tiempo, analisis_pib["Guerrero"],label="Guerrero")

plt.plot(tiempo, analisis_pib["Oaxaca"],label="Oaxaca")

plt.plot(tiempo, analisis_pib["Chiapas"],label="Chiapas")

plt.legend(loc="upper left")

plt.xlabel('Año')

plt.ylabel('PIB per cápita')

plt.title('Comportamiento del PIB per cápita de los más bajos')

plt.show()

fig,convergencia_bajos=plt.subplots()

convergencia_bajos.scatter(np.log(analisis_convergencia_bajos["y"]),analisis_convergencia_bajos["g_estimada"])

convergencia_bajos.set_xlabel("PIB per cápita inicial 2005")

convergencia_bajos.set_ylabel("Promedio de g 2005-2018")

plt.title('Convergencia de los más bajos')

plt.show()