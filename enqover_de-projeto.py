import math as m
import pandas as pd
import numpy as np
from scipy.optimize import minimize,fsolve
from scipy.integrate import odeint
from scipy.stats import linregress
def uloc(T):# graus em centigrados
    return (-16.68*10**-7)*T+40.30*10**-5
def ulxi(T):# graus em centigrados
    return (-17.95*10**-7)*T+50.59*10**-5
def ulaqua(T):
    return (1.24+0.014*T)*10**-3
def kloc(T):
    return (-2.95*10**-7*T)+1.37*10**-4
def klxi(T):
    return (-2.30*10**-7*T)+1.37*10**-4
#def func1p(x,y,z):# y representa a fração de octano, z representa a pressao em mmHg - falta meter o scipy a funcionar
    #return y*(10**(6.9187-(1351.76/(209.1+x))))+(1-y)*(10**(7.0015-(1476.39/(213.87+x))))-z
    
A_oct=6.9187
B_oct=1351.76
C_oct=209.10
A_xil=7.0015
B_xil=1476.39
C_xil=213.87
A_agua=8.071
B_agua=1730.60
C_agua=233.40



def pressao_vapor(T,A,B,C):
    eq=10**(A-B/(T+C))
    return eq


bar=100000 # 1 bar = 100000 pascal
mmHg=0.001333 # mmHg para Bar
T=82.05 # ªC
ploct=608.5  #kg/m3
plxil=783.7   #kg/m3
pagua=1000  # kg/m3
ulxil=ulxi(T) # Pas
uloct=uloc(T) #Pas
ulagua=ulaqua(T) #Pas
kloct=kloc(T)  # Kw/mK
klxil=klxi(T)  # Kw/mK
klagua=0.591 # Kw/mK
PM_oct=114.12 #kg/kmol
PM_xil=106.12 #kg/kmol
PM_agua=18 #kg/kmol
xoct=0.986
ulmistura=2.88e-4 #Pas
Qmolar_total= 251.01 #kmol/H # V
R=4
D=50.2 #kmol/h
W=Qmolar_total-D #kmol/h
Qdest_mass_oct=(D*PM_oct*xoct)/3600 # kg/s
Qdest_voct=Qdest_mass_oct/ploct #m3/s
Qdest_mass_xil=(D*PM_xil*(1-xoct))/3600 # kg/s
Qdest_vxil=Qdest_mass_xil/plxil #m3/s
Dmass=Qdest_mass_xil+Qdest_mass_oct
Dv=Qdest_vxil+Qdest_voct 
Qm_kgs=((Qmolar_total*PM_oct*xoct+Qmolar_total*PM_xil*(1-xoct))/3600)#kg/s
Qvoct=((Qm_kgs*xoct)/ploct) # m3/s
Qvxil=((Qm_kgs*(1-xoct))/plxil) #m3/s
Qv_total=Qvxil+Qvoct #m3/s
DeltaP_Caixa=0.005 #bar
P_dps_condensador=0.25-DeltaP_Caixa #bar
Lrv=Qv_total-Dv # refluxo
Qv_total
Lr=D*4
Lr
P_dps_condensador
PM_medio=PM_oct*(xoct)+PM_xil*(1-xoct)
pmedia=ploct*xoct+plxil*(1-xoct)
((PM_medio*Qmolar_total)/3600)/(pmedia)
paspen=5.7443 #kmol/m3
pmedio=paspen*PM_medio
pmedio

tempo_residencia_refluxo=5 # 5 minutos
V=tempo_residencia_refluxo*Qv_total*60
V_real_seguranca=2*V # multipliquei por 2 uma vez que nas heuristicas dizem que o tempo de residencia é para Half fll drums
D=round(((4*V_real_seguranca)/(3*m.pi))**(1/3),3) # m3
L=round(3*D,3)
print(f"D = {D} m and L={L} m")
V
g=9.81 
perda_carga_valvula_controlo=0.7 #bar
acessorios={"T":0.4,"Cotovelo=90° ell, standard":0.75,"gate vale/válvula de globo":0.17}

def vreal(d,q):
    vreal_tub=4*(q/((d*10**-3)**2*m.pi)) #com o Diametro em mm
    return vreal_tub
def area_corte_tubo(d):# d em mm
    area=(m.pi*(d*10**-3)**2)/4
    return area
def contracao(A1,A2):
    """ Obtencao do valor de K para a contração\n argumentos = A1, área do tubo de entrada e A2,area do tubo de saida"""
    K=0.5*(1-A2/A1)
    return K     
def expansao(v1,A1,A2):
    lc=((v1**2)/2)*(1-A1/A2)**2
    return lc
def perda_de_carga_acessorio(K,v):
    a= K*((v**2)/(2*g))
    return a
def perda_de_carga_linha(f,L,D,v):
    """ L em metros, Dimaetro em metros , velocidade em m/s"""
    return 4*f*(L/D)*(v**2)/(2*g)
''''def NPSHdisp(f,L,D,v,K,pcv,P1,p,z1,Pvt):
    a=perda_de_carga_acessorio(K,v)
    b=perda_de_carga_linha(f,L,D,v)
    DeltaHf=a+b+pcv
    NPSHdisp=(P1/(p*g))+z1-DeltaHf-Pvt/(p*g)
    return NPSHdisp'''
def delta_hbomba(z1,p1,z2,p2,v1,v2,deltah_atrito,p):
    return deltah_atrito+(z2-z1)+(p2/(p*g)-p1/(p*g))+(v2**2/(2*g)-v1**2/(2*g))
def potenciaf(n,Pu):
    # corresponde à potencia fornecida pela bomba
    return Pu/n
def validade(npsh,npshex):
    if npsh>npshex:
        return print(f"Não vai haver cavitação, uma vez que {round(npsh,2)}>{round(npshex,2)}")
    else:
        return print(f"Há cavitação!!!! Possível solução:Aumentar a altura !")

troco3={"tamanho":5+5,"cotovelos":2,"valvula de corte":2}
troco4={"tamanho":22.3-5,"cotovelos":4,"contracao":1,"expansao":1,"valvula de corte":2,"T como L":1}
troco5={"tamanho":110-5,"cotovelos":10,"contracao":2,"expansao":2,"valvula de corte":4,"T como L":1,"Permutador de Placas":1}
df3=pd.DataFrame(troco3,index=[3])
df4=pd.DataFrame(troco4,index=[4])
df5=pd.DataFrame(troco5,index=[5])
dff=pd.concat([df3,df4,df5])
dff["tamanho"][0:2].sum()
dff.fillna(0,inplace=True)                                            
v=2 #m/s
#Dados
#Caudal Volumétroc

Qv_total #m3/s
#velocidade / suposicao

v=2 #m/s

# Diametro obtido com a suposicao da velocidade 2m/s

Diam_tub3=m.sqrt(4*Qv_total/(m.pi*v))*1000 #mm

#Diâmetro comercial (tem de se tirar 2 espessuras) - Com o dimetro calculado acima foi-se ver o tubo comerical apropriado

Diam_com_tub3=(101.6+100)/2-4*2 # em mm

#velocidade real tubo 3

vreal_tub3=vreal(Diam_com_tub3,Qv_total) # m/s velocidade real do  caudal dentro do tubo 3<

# Tubo 4 - Caudal de refluxo(Dv) ,Tubo 5 - Caudal de destilado

Caudal_tubo4=Lrv

Caudal_tubo5=Dv

# Assumindo que o tubo4  tem o mesmo diâmetro que o tubo3.
# vreal do tubo 4 é:

vreal_tub4=vreal(Diam_com_tub3,Lrv) # m/s

#Calcular do diâmetro do tubo 5.Fazer o mesmo que para o tubo3.

#Assumindo v=2

Diam_tub5=m.sqrt(4*Caudal_tubo5/(m.pi*v))*1000 #mm

#Diâmetro comercial (tem de se tirar 2 espessuras) -Com o dimetro calculado acima foi-se ver o tubo comerical apropriado

Diam_com_tub5=(48.8+47.9)/2-3.2*2 # em mm

# Cálculo da velocidade no tubo 5

vreal_tub5=vreal(Diam_com_tub5,Dv) # m/s

A3=area_corte_tubo(Diam_com_tub3) #m2
A4=area_corte_tubo(Diam_com_tub3) #m2
A5=area_corte_tubo(Diam_com_tub5) # m2

df_d=pd.DataFrame({"velocidade (m/s)":[vreal_tub3,vreal_tub4,vreal_tub5],\
                   "Diametros (mm)":[Diam_com_tub3,Diam_com_tub3,Diam_com_tub5],"Areas de corte (m2)":[A3,A4,A5]},index=["Tubo3",\
                                                                                            "Tubo4","Tubo5"])

display(df_d)
Diam_tub5
Qv_total*3600
#Dados
#Caudal Volumétrico
Qv_total 

# Pressões

P3=P_dps_condensador*bar #Pa#  pressao no tanque de refluxo
P4=0.25*bar # pressao na coluna de destilação em pascal
P5=101325 # Pascal ( Assumindo que a pressão no tanque é a atmosférica)

# alturas 

z3=5+D/2 #m #altura do tanque de refluxo ao chão
z4=15 #m # altura do coluna de destilação ao chao
z5=2# m  

#dados das correntes

T=82.05 # Celsius
ploct=608.5  #kg/m3
plxil=783.7   #kg/m3
pmedio #kg/m3
#acessorios
acessorios={"T":0.4,"Cotovelo=90° ell, standard":0.75,"gatev":0.17}

# K3 e K4 e K5

cotovelos3=dff.loc[3]["cotovelos"]*acessorios["Cotovelo=90° ell, standard"]
cotovelos4=dff.loc[4]["cotovelos"]*acessorios["Cotovelo=90° ell, standard"]
cotovelos5=dff.loc[5]["cotovelos"]*acessorios["Cotovelo=90° ell, standard"]

valvulas_corte3=dff.loc[3]["valvula de corte"]*acessorios["gatev"]
valvulas_corte4=dff.loc[4]["valvula de corte"]*acessorios["gatev"]
valvulas_corte5=dff.loc[5]["valvula de corte"]*acessorios["gatev"]

T_como_L=dff.loc[4]["T como L"]*acessorios["T"]
T_como_L5=dff.loc[5]["T como L"]*acessorios["T"]


K3=cotovelos3+valvulas_corte3
K4=cotovelos4+valvulas_corte4+T_como_L
K5=cotovelos5+valvulas_corte5+T_como_L5

# Perda de carga acessorios

Deltap_vc=(0.7*bar)/(g*pmedio) #m # perda de carga na válvula de controlo corrente 4 (valor tem de estar entre 0.7-0.8 bar)


'''perda de carga no permutador de placas,corrente 5 (valor tem de estar entre 0.1  e 0.5 bar.
Uma vez que so ha um permutador apenas somei isso a perda de carga.'''

Permuatador_pc=(0.3*bar)/(g*pmedio) #m 

expansao_tanque5=expansao(vreal_tub5,A5,10000000000)/g # o A2 dizemos que é muito grande

contracao5=contracao(A3,A5)  # entrada do tubo3 com maior Diametro para o tubo5 com menor diametro.

#Atanque=L*D # Considerando a area frontal de um cilindro em queda, como fazemos em OSM

Contracao_tanque_tubo3=contracao(1000000000000,A3) # Sendo que se podia até dizer que ATanque>>A4 entao..ignora-se o termo A2/A1
 
#Perda_carga de acessórios

perda_de_carga_acessorios34=perda_de_carga_acessorio(K3,vreal_tub3)+perda_de_carga_acessorio(K4,vreal_tub4)+ Deltap_vc\
                                +Contracao_tanque_tubo3#m
perda_de_carga_acessorios35=perda_de_carga_acessorio(K3,vreal_tub3)+perda_de_carga_acessorio(K5,vreal_tub5)+Permuatador_pc\
                            +contracao5+Deltap_vc+Contracao_tanque_tubo3 +expansao_tanque5 # m

perda_de_carga_acessorios4=perda_de_carga_acessorio(K4,vreal_tub4)+Deltap_vc
perda_de_carga_acessorios5=perda_de_carga_acessorio(K5,vreal_tub5)+Permuatador_pc\
                            +contracao5+Deltap_vc


#Obtenção de f ( o valor de rugosidade tirei do pwp do professor)

e=0.045 # mm # rugosidade 
Re3=(vreal_tub3*pmedio*Diam_com_tub3*10**-3)/ulmistura #número de Reynolds do Tubo3
Re4=(vreal_tub4*pmedio*Diam_com_tub3*10**-3)/ulmistura #número de Reynolds do Tubo4
Re5=(vreal_tub5*pmedio*Diam_com_tub3*10**-3)/ulmistura #número de reynolds do Tubo5
f3=0.0178
f4=0.0180
f5=0.0208

# Perda de carga de linha

perda_de_carga_troco3=perda_de_carga_linha(f3,dff.loc[3]["tamanho"],Diam_com_tub3*10**-3,vreal_tub3)
perda_de_carga_troco4=perda_de_carga_linha(f4,dff.loc[4]["tamanho"],Diam_com_tub3*10**-3,vreal_tub4)
perda_de_carga_troco5=perda_de_carga_linha(f5,dff.loc[5]["tamanho"],Diam_com_tub5*10**-3,vreal_tub5)

perda_de_carga_linha34=perda_de_carga_troco3+perda_de_carga_troco4 #m
perda_de_carga_linha35=perda_de_carga_troco3+perda_de_carga_troco5 #m

#Perda de carga de atrito total

perda_carga34=perda_de_carga_linha34+perda_de_carga_acessorios34

perda_carga35=perda_de_carga_linha35+perda_de_carga_acessorios35

# Deltah_bomba 3-4, 3-5

delta_h_bomba34=delta_hbomba(z3,P3,z4,P4,vreal_tub3,vreal_tub4,perda_carga34,pmedio)
delta_h_bomba35=delta_hbomba(z3,P3,z5,P5,vreal_tub3,vreal_tub5,perda_carga35,pmedio)

# Deltah_bomba 3-4, 3-5 com Fator de segurança de 10%

delta_h_b34seguro=delta_h_bomba34*1.1
delta_h_b35seguro=delta_h_bomba35*1.1



dic={"K3 (m)":K3,"K4 (m)":K4,"perda_de_carga_acessorios34 (m) ":perda_de_carga_acessorios34,\
"Re3":Re3,"Re4":Re4,"Diam_tub3 (m)":Diam_tub3,"perda_carga34 (m)":perda_carga34,"delta_h_bomba34 (m)":delta_h_bomba34}
df=pd.DataFrame(dic,index=["Troco 3- 4"])
df
perd=pd.DataFrame([delta_h_b34seguro,delta_h_b35seguro],index=["troco 3-4,DeltaH(m)","troco3-5,DeltaH(m)"])
perd.rename(columns={0:"Delta_H_fato de segurança (m)"})

Delta_H_bomba=max([delta_h_b35seguro,delta_h_b34seguro]) #m 
Qvtot_m3h=(Qv_total)*3600  # caudal volumetrico total em m3/h

valors=pd.Series([Delta_H_bomba,Qvtot_m3h],index=["Delta_H_bomba","Qvtot_m3h"])
x=pd.concat([perd,valors])
display(x.rename(columns={0:"dados"}))
delta_h_bomba35*1.1
z3
Delta_H_bomba=78 #metros

# DeltaP da bomba
Delta_P=Delta_H_bomba*pmedio*g

# Cálculo da Potência associado ao fluido
Potencia_fluido=Delta_P*Qv_total  # Joule/s, # Qv_total em m3/s

# eficiência iso.. global
eficiencia=0.57

#Potência do motor
Potencia_motor=Potencia_fluido/eficiencia
print(f" Potência do motor é de : {round(Potencia_motor,3)} J/s") #Joule /s
Potencia_fluido
perda_de_cargatroco3=perda_de_carga_acessorio(K3,vreal_tub3)+perda_de_carga_troco3+Contracao_tanque_tubo3
pressao_vapor_oct=(pressao_vapor(T,A_oct,B_oct,C_oct)*mmHg*bar*xoct)/(ploct*g) ## m
pressao_vapor_xil=(pressao_vapor(T,A_xil,B_xil,C_xil)*mmHg*bar*(1-xoct))/(plxil*g) ## m
pressao_media=pressao_vapor_oct*xoct+pressao_vapor_xil*(1-xoct)

#definir funcao para o NPSHdisp
def npshd(P1,p,DeltaH_admissao_troco_até_a_bomba,pv,z1):
    equacao=P1/(p*g)+z1-DeltaH_admissao_troco_até_a_bomba-pv/(p*g)
    return equacao

# NPSH exigido
NPSHexi=3 #m
#NPSHdisp
Npshdisp=npshd(P3,ploct,perda_de_cargatroco3,pressao_media,z3)
validade(Npshdisp,NPSHexi)
pressao_media
'''
Dimenções do equipamento sob vácuo:
Tanque de refluxo( D,L)
Coluna:
    D = 2.7
    L = 10
Condensador:
    D = 0.7874
    L = 4.33
    
'''
#tanque de refluxo
t_refluxo = [D,L]
#Coluna
col_dest = [2.7,10]
#Condensador
p_calor = [0.7874,4.33]

def vol_cilindro(D,L):
    return np.pi/4*D**2*L

def vol(equipamento):
    return vol_cilindro(equipamento[0],equipamento[1])

#considerou-se o ebulidor com 3*o tamanho do condensador e o volume dos tubos de 1m3
V_equipamento_sob_vácuo = vol(t_refluxo)+vol(col_dest)+vol(p_calor)+3*vol(p_calor)+1 

V_equipamento_sob_vácuo_feet = V_equipamento_sob_vácuo*35.3147

V_equipamento_sob_vácuo_feet
Q_ar_fugas_pounds = 38 #pound/h
Q_ar_fugas = Q_ar_fugas_pounds*0.453592/3600 #kg/s
Q_ar_fugas
#Qmolar_total


def solubilidade_o2(T):#ºC
    return np.e**(-6.23+0.219*100/(T+273))

PM_ar = 28.9 #kg/kmol

#######################mol/mol
s_n2_oct = 13.11*10**-4
s_o2_oct = 21.20*10**-4
s_n2_xil = 6.12*10**-4
s_o2_xil = 11.18*10**-4


T=40 # temperatura de saida do condensador

Qm_oct = Qm_xil = 50 #kmol/h

n_ar = (Qm_oct*(s_o2_oct+s_n2_oct)+Qm_xil*(s_o2_xil+s_n2_xil))/3600 #kmol/s

Q_ar_dissolvido = n_ar*PM_ar #kg/


Q_ar_total = Q_ar_dissolvido + Q_ar_fugas

P_v = (pressao_vapor(T,A_oct,B_oct,C_oct)*xoct+pressao_vapor(T,A_xil,B_xil,C_xil)*(1-xoct))*mmHg

#print(P_v,0.25*bar)

fracao_v_ar= P_v/(0.25-P_v)*PM_oct/PM_ar # massa de vapor / massa de ar ( Multiplicar pelo peso medio molecular)

Qv_vapor_arrastado=Q_ar_total*fracao_v_ar # massa de vapor arrastada kg/s

Qv_total=(Qv_vapor_arrastado+Q_ar_total) # kg/s

#print(Q_ar_dissolvido,Q_ar_total,Q_vapor_arrastado)
display(fracao_v_ar)

#Uma vez que se tem uma pressão absoluta de 0.25bar só se precisa de um ejetor
if P_dps_condensador/mmHg>50:
    print("Um ejetor")
    
Q_ar_total

#Valores arbitrados
%precision 4

mmHg1=750.062 # bar para mmHg
Po3=1 # bar
Poa=8# bar #pressão de vapor a entrar no ejetor
T=40  # temperatura a que sai os gass nao condensaveis do condensador


# Valores sabidos

Pob=P_dps_condensador

# Obtencao do Peso molecular do vapor arrastado

# Calculo do caudal molar de vapor arrastado
Qvapor_mol=Qv_vapor_arrastado/(PM_medio*1e-3)

# Calculo do caudal molar de ar que se encontra no vapor arrastado
Qvar_mol=Q_ar_total/(PM_ar*1e-3)

#Calculo do caudal total molar que sai do evaporador na forma gasosa

Qv_sai_evap=Qvar_mol+Qvapor_mol

# Fração molar de ar  da corrente gasosa
xar=Qvar_mol/(Qv_sai_evap)

#Fracao molar de octano+xileno que sai da corrente gasosa

xoct_xil=(1-xar)

PM_vapor_arrastado=xar*PM_ar+(1-x)*PM_medio # peso molecular da corrente que sai do condensador por cima

P_dps_condensador # pressao depois do condensador




#### Definicao da funcao a partir do qual se obtem o valor de wa_corrigido

def correcao(wa,wb,toa,tob,ma,mb):
    x=wb/wa*np.sqrt((toa*ma)/(tob*mb)) # onde o x corresponde à fração corrigda
    return x


# Razao de compresão
pobpoa=Pob/Poa
po3pob=Po3/Pob

# Atreavés do gráfico do Perry, consegue-se otbter

wbwa=0.022

wa=Qv_total/wbwa # onde qv_total corresponde ao caudal que é arrastado pelo vapor wa ( estando este em kg/s)

#gama_vapor=1.31

# Correção do wa,uma vez que se considera que a temperatura entre o vapor arrastado e o vapor que arrasta é diferente.

#fracao_real_wbwa=correcao(wa,Qv_total,T_agua+273.15,T+273.15,PM_agua,PM_vapor_arrastado)

#wa1_cor=Qv_total/fracao_real_wbwa # quantidade de vapor a ser fornecido ao ejetor em kg/s

#Qec1=wa_cor+Qv_total # quantidade de vapor que entra no condensador

d={"Po3/Pob":po3pob,"Pob/Poa":pobpoa}
df1=pd.DataFrame(d,index=[0])
df1
# Razao de compresão
pobpoa=Pob/Poa
po3pob=Po3/Pob

# Atreavés do gráfico do Perry, consegue-se otbter

wbwa=0.022

wa=Qv_total/wbwa # onde qv_total corresponde ao caudal que é arrastado pelo vapor wa ( estando este em kg/s)

#gama_vapor=1.31

# Correção do wa,uma vez que se considera que a temperatura entre o vapor arrastado e o vapor que arrasta é diferente.

#fracao_real_wbwa=correcao(wa,Qv_total,T_agua+273.15,T+273.15,PM_agua,PM_vapor_arrastado)

#wa1_cor=Qv_total/fracao_real_wbwa # quantidade de vapor a ser fornecido ao ejetor em kg/s

#Qec1=wa_cor+Qv_total # quantidade de vapor que entra no condensador

d={"Po3/Pob":po3pob,"Pob/Poa":pobpoa}
df1=pd.DataFrame(d,index=[0])
df


# Condensador  barométrico
# interessa saber a sua altura por causa do layout
# saber a quantidade de agua necessaria,uma vez que a agua é cara.
#Quantidade de vapor no ejetor
#altura do condensador barometrico

### https://webbook.nist.gov/cgi/cbook.cgi?ID=C111659&Mask=4,vaporizacao do octano.
##Cps: N-octano (0-50) : 0,505 cal/g (20-123): 0,572 cal/g
## cps: Xilema(30): 0,411 cal/g (39): 0,450 cal/g
##

# Calculo da agua necessaria : Balanco entalpico
Te_agua_cb1=27
Ts_agua_cb1=35

#Pressão à saida do 1 condensador e à entrada do 2 ejetor
P2=np.sqrt(P_saida*0.25) # corrigir ainda esta pressão de 0.25..

## https://preview.irc.wisc.edu/properties/ para obtencao das propriedades da agua
# Estado de referencia : T: 0,01ºC , água liquida saturada,P=0.611 kPa

He_agua_c=113000 # J/kg água liquida de entrada a 27 Graus
Hs_agua_c=147000 # J/kg água liquida de saida a 40 graus celsius.
Hve_agua_c=2839.8e3 # J/kg Vapor à pressão de 8 bar superaquecido




# Pressuposto para calcular wb à saida do condensador -> Wa_cor é igual quer seja para o 1 quer seja..para o 2 ejetor.
# wa_cor = vapor.Desprezei nao condensaveis.Para o balanco entalpico estou a desprezar o wb
def funcao(x):
    erro=(wa_cor*Hve_agua_c+x*He_agua_c-(x+wa_cor)*Hs_agua_c)**2
    return erro*1e10

Qagua1=minimize(funcao,1,method="Nelder-Mead",tol=1e-5).x[0] # kg/s caudal de água para o primeiro ejetor

Qagua1 # kg/s caudal de agua


P_dps_condensador




