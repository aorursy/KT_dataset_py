# heating the engines
%matplotlib inline

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib import rcParams
# import mpl_toolkits
# import math
from math import exp, log, log1p
from random import random
from IPython.display import display
from IPython.display import HTML
import seaborn as sns
## Function to highlight values at the top of the plotted bars
#
# usage (rects -> bar,  ax -> axe, cor -> color, alt -> 0.90 means 90% of the height of the bar)
# begin def func
def autolabel(rects, ax, cor= 'white', alt=0.90):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height*alt,
                '%d' % int(height),
                ha='center', va='bottom', fontsize=10, color=cor, fontweight='bold' )
# end def        
        
# Loads the dataframe
data = pd.read_csv("../input/real-state-data-rio-brasil/z2017-mod-entrada.csv", delimiter=";", encoding="cp1252", skiprows=1)
display(data.head(10))
display(data.describe())
data = pd.read_csv("../input/real-state-data-rio-brasil/z2017-mod-entrada.csv", delimiter=";", encoding="cp1252", skiprows=1)
display(data.head(10))
# sumary of descriptive statistics
data = pd.read_csv("../input/real-state-data-rio-brasil/z2017-mod-entrada.csv", delimiter=";", encoding="cp1252", skiprows=1)
display(data['preco'].describe().apply(lambda x: format(x, 'f')))
# This piece of code is far from optimized. Please be patient
# data = pd.read_csv("z2017-mod-entrada.csv", delimiter=";", encoding="cp1252", skiprows=1)
#
# Frequency - Preco
fig2 = plt.figure(figsize=(16, 5))
ax = plt.subplot(132)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
# Color patternss
cm = plt.cm.get_cmap("Purples")
# Dataframe
X = data["preco"].value_counts(sort=False).index.tolist()
y = data["preco"].value_counts(sort=False).tolist()
yval = y
xval = np.log1p(X)
tam = len(xval)/10
cor = cm(np.multiply(xval,1/7))
# Axes limits
plt.xlim(11,17)
plt.ylim(0,55)
# Plot 2
matplotlib.rc('font', family='DejaVu Sans')
plt.bar(xval, yval, linewidth=0, color=cor, edgecolor='black', alpha=0.55, width=0.1);
plt.title (u'Distribution log of Precos (prices)', fontsize=16, fontweight='bold')
plt.ylabel(u'Frequency', fontsize=10, fontweight='bold') 
plt.xlabel(u'Histogram preco (log)', fontsize=10)
# 'preco' labels 
sxval = sorted(xval)
# Customize legend colors
cores=[]
cores.append(cm(np.multiply(sxval[2],0.2/7)))
cores.append(cm(np.multiply(sxval[81],0.25/7)))
cores.append(cm(np.multiply(sxval[203],0.3/7)))
cores.append(cm(np.multiply(sxval[304],0.35/7)))
cores.append(cm(np.multiply(sxval[365],0.4/7)))
cores.append(cm(np.multiply(sxval[376],0.45/7)))
# Legend creation
tam = len(cores)
b_patch = []
labels = ['0 a 200 mil', '201 a 500 mil', '501 a 1.0 mil', '1.1 a 2.0 mil', '2.1 a 5.0 Mi', '+ de 5.0 Mi']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
plt.legend(handles=b_patch, edgecolor='black')
# Plot 1
ax7 = plt.subplot(131) 
fig2.add_subplot(ax7) 
sns.distplot(data['preco'], axlabel="Preço", kde_kws={"color": "purple", "lw": 1, "label": "Cauda à direita"});
plt.title (u'Distribution of Precos', fontsize=16, fontweight='bold')
# Sources and authoring
plt.text(11, -8, "source: real state announces "  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold', clip_on=True)
# Plot 3
sns.set(color_codes=True)
ax7 = plt.subplot(133) 
fig2.add_subplot(ax7) 
colors = plt.cm.plasma(np.log1p(data["preco"].fillna(0).values))
sns.distplot(xval, axlabel="Preço log (BRL)", color="r", kde_kws={"color": "black", "lw": 2, "label": "equilibrada"});
plt.title (u'Distribution of log Preços', fontsize=16, fontweight='bold')
# Sources and authoring
plt.text(11, -8, "source: real state announces "  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold', clip_on=True)
plt.show();

# skewness e kurtosis
print(" Assimetria (skewness): %f" % data["preco"].skew())
print("Achatamento (kurtosis): %f" % data["preco"].kurt())
# Return the log(1+X) for every cell in the dataset
sk = np.log1p(data["preco"]).skew()
kt = np.log1p(data["preco"]).kurt()
# skewness e kurtosis
print(" Assimetria (skew): "+"{:< 10f}".format(sk))
print("Achatamento (kurt): "+"{:< 10f}".format(kt))
# stats sumary 
display(data['condominio'].describe().apply(lambda x: format(x, 'f')))
# This piece of code is far from optimized. Please be patient.
# Feature analysis
# Freq - condominio
#
# Graph - 1 No log values (Raw data)
# Define font to the two plots
matplotlib.rc('font', family='DejaVu Sans')
#Figure creation
fig = plt.figure(figsize=(16, 5))
ax0 = plt.subplot(121)  
fig.subplots_adjust(hspace=0.5)
fig.add_subplot(ax0)
ax0.spines["top"].set_visible(False)  
ax0.spines["right"].set_visible(False)  
ax0.get_xaxis().tick_bottom()  
ax0.get_yaxis().tick_left()  
cm = plt.cm.get_cmap('viridis_r')
matplotlib.rc('font', family='DejaVu Sans')
targ = data["condominio"].value_counts(sort=False)
X = data["condominio"].value_counts(sort=False).index.tolist()
yval = list(targ)
xval= X
plt.xlim(0,8000)
plt.ylim(0,90)
cor = cm(np.multiply(xval,1/9))
# histogram
rects = plt.bar(xval, yval, linewidth=0.8, edgecolor='black', color=cor, alpha=1, width=10);
label = 'Quotas Condominiais'
plt.title (label, fontsize=16, fontweight='bold')
plt.ylabel('Frequência', fontsize=14, fontweight='bold') 
plt.xlabel('Valores em BRL', fontsize=14, fontweight='bold')
plt.text(0, -15, "Fonte: anúncios de imóveis - RJ dez/2017 | "  
"By A. R. Barberá", fontsize=10, color='black', fontweight='bold')
# Graph 2 - Log values
plt.tight_layout()
ax1 = plt.subplot(122)  
fig.subplots_adjust(hspace=0.5)
fig.add_subplot(ax1)
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left()  
cm = plt.cm.get_cmap('viridis_r')
# Define font 
matplotlib.rc('font', family='DejaVu Sans')
targ = data["condominio"].value_counts(sort=False)
X = np.log1p(data["condominio"].value_counts(sort=False).index.tolist())
yval = list(targ)
xval= X
plt.xlim(0,10)
plt.ylim(0,90)
cor = cm(np.multiply(xval,1/9))
# Histogram
rects = plt.bar(xval, yval, linewidth=1, color=cor, edgecolor='black', alpha=1, width=0.6);
label = 'Quotas Condominiais'
plt.title (label, fontsize=16, fontweight='bold')
plt.ylabel('Frequência', fontsize=14, fontweight='bold') 
plt.xlabel('Valores em log (BRL)', fontsize=14, fontweight='bold')
# Legend colors and labels 
cores=[]
cores.append(cm(np.multiply(xval[1],1/18)))
cores.append(cm(np.multiply(xval[14],1/14)))
cores.append(cm(np.multiply(xval[24],1/12)))
cores.append(cm(np.multiply(xval[48],1/10)))
cores.append(cm(np.multiply(xval[92],1/9)))
cores.append(cm(np.multiply(xval[122],1/8)))
cores.append(cm(np.multiply(xval[342],1/7)))
cores.append(cm(np.multiply(xval[381],1/6)))
cores.append(cm(np.multiply(xval[400],1/5)))
tam = len(cores)
b_patch = []
labels = ['120-300', '301-600', '601-900', '901-1200', '1201-1500', '1501-1800', '1801-2100', '2101-2400', '+ 2400']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
plt.legend(handles=b_patch, edgecolor='black')
plt.show();
# This piece of code is far from optimized. Please be patient.
#
# This function put the y value at the top-in the bars
#
# Begin graph        
# Fig mount        
fig0 = plt.figure(figsize=(14, 5))        
ax2 = plt.subplot(121)  
fig0.add_subplot(ax2)
ax2.spines["top"].set_visible(False)  
ax2.spines["right"].set_visible(False)  
ax2.get_xaxis().tick_bottom()  
ax2.get_yaxis().tick_left() 
plt.xlim(0.5,3.5)
plt.ylim(0,800)
# Colors 1st graph
cm = plt.cm.get_cmap("Greens")
matplotlib.rc('font', family='DejaVu Sans')
targ = data["quartos"].value_counts(sort=False).index.tolist()
val1 = data["quartos"].value_counts(sort=False).tolist()
yval =  val1[:]
xval = [1,2,3]
cor = cm(np.multiply(xval,2/len(xval)))
# Histogram
rects = plt.bar(xval, yval, linewidth=1.5, color=cor, edgecolor='black', alpha=0.65, width=0.5);
# Titles and font
plt.title ('Distribution of number of rooms', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold') 
plt.xlabel('Num. of rooms', fontsize=14, fontweight='bold')
plt.text(0.5, -140, "source: real state announces |"  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold')  
# Highlights bar values
autolabel(rects, ax2, 'cyan',0.9);
#
# Graph 2
ax8 = plt.subplot(122) 
fig0.add_subplot(ax8) 
# Get percents
freqs = data["quartos"].value_counts(sort=False)
tam = len(freqs)
soma = 0
for i in range(tam):
    soma = soma + freqs.iloc[i]
fracs = []    
for i in range(tam):
    frac = (freqs.iloc[i]/soma)*100
    fracs.append(frac)
# Get labels
labels = '1r', '2r', '3r'
explode = (0, 0.05, 0)
# Make square figures and axes
patches, texts, autotexts = plt.pie(fracs, explode=explode,
                                    labels=labels, autopct='%.0f%%',
                                    shadow=True, radius=1.0)
for t in texts:
    t.set_size('small')
for t in autotexts:
#    t.set_size('x-small')
    t.set_size('small')
autotexts[0].set_color('w')
plt.title ('Distribution percentual of real state, by number of rooms', fontsize=14, fontweight='bold')
plt.xlabel('% by type (1, 2 e 3rooms)', fontsize=14, fontweight='bold')
plt.show();
# This piece of code is far from optimized. Please be patient.
#
# >>>>>>>>>>>>>>>>>>>>>>>>>
# Freq. number of suites
#
# Dimensoes da figura e limites dos eixos
fig = plt.figure(figsize=(16, 10))
#plt.tight_layout()
ax3 = plt.subplot(221) 
fig.add_subplot(ax3) 
ax3.spines["top"].set_visible(False)  
ax3.spines["right"].set_visible(False)  
ax3.get_xaxis().tick_bottom()  
ax3.get_yaxis().tick_left() 
plt.xlim(-0.5, 4);
plt.ylim(0, 2000);
# Define color palette and value
cm = plt.cm.get_cmap("Blues_r")
targ = data["suites"].value_counts(sort=False).index.tolist()
val1 = data["suites"].value_counts(sort=False).tolist()
yval =  val1[:]
xval = [0,1,2,3]
cor = cm(np.multiply(xval,0.8/len(xval)))
# Plota o histograma
rects = plt.bar(xval, yval, linewidth=1, color=cor, edgecolor='black', alpha=0.75, width=0.75);
plt.title ('Distribution of number of suites', fontsize=14, fontweight='bold');
plt.ylabel('Frequency', fontsize=14, fontweight='bold');
plt.xlabel('Num. of suítes', fontsize=14, fontweight='bold');
# Identifica a fontes dos dados e autoria
# plt.text(-0.5, -200, "Fonte: sites de anúncios de imóveis | "  
#        "Autor: Roberto Barberá", fontsize=10, fontweight='bold')  
# Highlights...
autolabel(rects, ax3, 'blue', 1.015)
#
# Freq - Num. de vagas
#
# Fig dimentions and axes limits
# 
ax4 = plt.subplot(222)  
fig.add_subplot(ax4)
ax4.spines["top"].set_visible(False)  
ax4.spines["right"].set_visible(False)  
ax4.get_xaxis().tick_bottom()  
ax4.get_yaxis().tick_left() 
plt.xlim(-0.5, 5);
plt.ylim(0, 2000);
# Define color palette
cm = plt.cm.get_cmap("BuGn_r")
targ = data["vagas"].value_counts(sort=False).index.tolist()
val1 = data["vagas"].value_counts(sort=False).tolist()
yval =  val1[:]
xval = [0,1,2,3,4]
cor = cm(np.multiply(xval,0.8/len(xval)))
# Plota o histograma
rects = plt.bar(xval, yval, linewidth=1, color=cor, edgecolor='black', alpha=0.75, width=0.75);
plt.title ('Distribution of number of vagas', fontsize=14, fontweight='bold');
plt.ylabel('Frequency', fontsize=14, fontweight='bold');
plt.xlabel('Num. of vagas', fontsize=14, fontweight='bold');
# plt.text(-0.5, -200, "Fonte: sites de anúncios de imóveis | "  
#        "Autor: Roberto Barberá", fontsize=10, fontweight='bold')  
# Highlights values on bars
autolabel(rects, ax4, 'red', 1.015)
# Frequency of Area
#
ax5 = plt.subplot(223) 
fig.add_subplot(ax5) 
ax5.spines["top"].set_visible(False)  
ax5.spines["right"].set_visible(False)  
ax5.get_xaxis().tick_bottom()  
ax5.get_yaxis().tick_left()  
plt.xlim(0,480);
plt.ylim(1,90);
#  Defines and values
cm = plt.cm.get_cmap("Greens")
targ = data["area"].value_counts(sort=False).index.tolist()
val1 = data["area"].value_counts(sort=False).tolist()
yval = val1
xval = targ
cor = cm(np.multiply(xval,0.8/len(xval)))
# Plots histogram
rects = plt.bar(xval, yval, linewidth=0.5, color=cor, edgecolor='black', alpha=0.75, width=10);
# Define titles, font e authoring
plt.title ('Distribution of areas (m2)', fontsize=14, fontweight='bold');
plt.ylabel('Frequency', fontsize=14, fontweight='bold');
plt.xlabel('Area (m2)', fontsize=14, fontweight='bold');
plt.text(10, -20, "source: real state announces | "  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold');
# Legends and prices
sxval = sorted(xval)
cores=[]
#cores.append(cm(np.multiply(xval[330],1/num_faixas_leg)))
cores.append(cm(np.multiply(sxval[0],1/170)))
cores.append(cm(np.multiply(sxval[30],1/240)))
cores.append(cm(np.multiply(sxval[60],1/250)))
cores.append(cm(np.multiply(sxval[93],1/250)))
cores.append(cm(np.multiply(sxval[130],1/250)))
cores.append(cm(np.multiply(sxval[180],1/450)))
# Create the legend   
tam = len(cores)
b_patch = []
labels = ['20 a 50m2', '51 a 80m2', '81 a 110m2','111 a 150m2', '151 a 300m2', '+ de 300m2']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
plt.legend(handles=b_patch, edgecolor='black')

#-----------------------------------------------> plt.show()
# Freq - Bairro
#
ax6 = plt.subplot(224) 
fig.add_subplot(ax6) 
ax6.spines["top"].set_visible(False)  
ax6.spines["right"].set_visible(False)  
ax6.get_xaxis().tick_bottom()  
ax6.get_yaxis().tick_left()  
# Color settings and Data
cm = plt.cm.get_cmap("copper_r")
targ = data["bairro"].value_counts(sort=False).index.tolist()
val1 = data["bairro"].value_counts(sort=False).tolist()
yval =  val1[:]
xval = targ[:]
# Axes limits 
plt.xlim(0,8.5);
plt.ylim(0,360);
cor = cm(np.multiply(xval,0.8/len(xval)))
# Bar chart
rects = plt.bar(xval, yval, linewidth=1, color=cor, edgecolor='black', alpha=1, width=0.55);
plt.title ('Apartments by bairro (neighborhooed)', fontsize=14, fontweight='bold');
plt.ylabel('Frequency', fontsize=14, fontweight='bold');
plt.xlabel('Bairro', fontsize=14, fontweight='bold');
# Legends for places (in Brasil we don't have counties)
cores=[]
for i in xval:
    cores.append(cm(np.multiply(i,0.8/len(xval))))
tam = len(cores)
b_patch = []
labels = ['Bot', 'Cop', 'Gav', 'Gra', 'Ipa', 'Leb', 'Tij']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
# Legends constructor and authoring
plt.legend(handles=b_patch, edgecolor='black')
#
# Plot height values inside and at the top of the bars, with diferent colors and an offset multiplier. 
# Usage: autolabel(rects,color,offset) // default color='white' and default offset=0.90
autolabel(rects,ax6,'white',0.90)
plt.show();
# This piece of code is far from optimized. Please be patient.
#
# Freq - Area
# Fig dim and axes limits
fig = plt.figure(figsize=(16, 5))
plt.tight_layout()
ax5 = plt.subplot(121) 
fig.add_subplot(ax5) 
ax5.spines["top"].set_visible(False)  
ax5.spines["right"].set_visible(False)  
ax5.get_xaxis().tick_bottom()  
ax5.get_yaxis().tick_left()  
plt.xlim(0,10);
plt.ylim(0,90);
# Define color palette and values
cm = plt.cm.get_cmap("Greens")
targ = data["area"].value_counts(sort=False).index.tolist()
val1 = data["area"].value_counts(sort=False).tolist()
yval = val1
xval = np.log1p(targ)
cor = cm(np.multiply(np.square(xval),1/50))
# Plot
rects = plt.bar(xval, yval, linewidth=0.5, color=cor, edgecolor='black', alpha=0.75, width=0.5);
# Define titles, fonts and authoring
plt.title ('Distribution of area', fontsize=14, fontweight='bold');
plt.ylabel('Frequency', fontsize=14, fontweight='bold');
plt.xlabel('Área log (m2)', fontsize=14, fontweight='bold');
plt.text(0, -16, "source: real state announces | "  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold');
# Legends
sxval = sorted(xval)
# Get color
cores=[]
cores.append(cm(np.multiply(sxval[0],1/10)))
cores.append(cm(np.multiply(sxval[30],1/9)))
cores.append(cm(np.multiply(sxval[60],1/8)))
cores.append(cm(np.multiply(sxval[93],1/8)))
cores.append(cm(np.multiply(sxval[130],1/8)))
cores.append(cm(np.multiply(sxval[180],1/8)))
# Create legend labels   
tam = len(cores)
b_patch = []
labels = ['20 a 50m2', '51 a 80m2', '81 a 110m2','111 a 150m2', '151 a 300m2', '+ de 300m2']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
plt.legend(handles=b_patch, edgecolor='black')
#
# Freq - Tempo de anuncio (Atualizacao)
#
ax = plt.subplot(122)  
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
# Plot patterns 
cm = plt.cm.get_cmap("YlGnBu")
targ = data["atualizacao"].value_counts(sort=False).index.tolist()
val1 = data["atualizacao"].value_counts(sort=False).tolist()
yval = np.log1p(val1)
xval = np.log1p(targ)
xval = targ
cor = cm(np.multiply(xval,1/20))
# Axes limits
plt.xlim(-0.5,32)
plt.ylim(0,8)
# Bar graph
matplotlib.rc('font', family='DejaVu Sans')
plt.bar(xval, yval, linewidth=1, color=cor, edgecolor='black', alpha=0.75, width=0.6)
plt.title (u'Distribution of time of advertise', fontsize=14, fontweight='bold')
plt.ylabel(u'Frequency', fontsize=14, fontweight='bold') 
plt.xlabel(u'Time of advertise (days)', fontsize=14, fontweight='bold')
# Legend labels
cores=[]
cores.append(cm(np.multiply(sxval[2],0.05)))
cores.append(cm(np.multiply(sxval[11],1/10)))
cores.append(cm(np.multiply(sxval[27],1/6)))
cores.append(cm(np.multiply(sxval[28],1/5)))
cores.append(cm(np.multiply(sxval[39],1/5)))
sxval = sorted(xval)
# Created legend
tam = len(cores)
b_patch = []
labels = ['0 a 1d', '2 a 10d', '11 a 30d', '31 a 1ano', '+ de 1ano']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
plt.legend(handles=b_patch)
plt.show();
# skewness e kurtosis
print(" Assimetria (skewness): %f" % np.log1p(data["area"]).skew())
print("Achatamento (kurtosis): %f" % np.log1p(data["area"]).kurt())
# Freq - Distância
#
# Fig dimensions ans graph limits
fig = plt.figure(figsize=(14,7))
ax = plt.subplot(111)
fig.add_subplot(ax)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
# Graph patterns
cm = plt.cm.get_cmap("YlGnBu")
targ = data["distancia"].value_counts(sort=False).index.tolist()
val1 = data["distancia"].value_counts(sort=False).tolist()
yval = np.log1p(val1)
xval = np.log1p(targ)
cor = cm(np.multiply(xval,1/7))
# Axes limits
plt.xlim(0,8)
plt.ylim(0,8)
# Bar graph
matplotlib.rc('font', family='DejaVu Sans')
plt.bar(xval, yval, linewidth=1, color=cor, edgecolor='black', alpha=0.75, width=0.3)
plt.title (u'Distribution of distances', fontsize=14, fontweight='bold')
plt.ylabel(u'Frequency (log)', fontsize=14, fontweight='bold') 
plt.xlabel(u'Distance log (km)', fontsize=14, fontweight='bold')
plt.text(0, -1.3, "source: real state announces | "  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold');
# Legends of prices
sxval = sorted(xval)
num_faixas_leg = 7
# Legend colors 
cores=[]
cores.append(cm(np.multiply(sxval[58],1/7)))
cores.append(cm(np.multiply(sxval[137],1/7)))
cores.append(cm(np.multiply(sxval[154],1/7)))
cores.append(cm(np.multiply(sxval[160],1/7)))
cores.append(cm(np.multiply(sxval[164],1/7)))
cores.append(cm(np.multiply(sxval[175],1/7)))
cores.append(cm(np.multiply(sxval[178],1/9.2)))
# Create legend   
tam = len(cores)
b_patch = []
labels = ['0 a 0.5km', '0.6 a 1km', '1 a 2km','2.1 a 2.5km','2.6 a 3km', '3.1 a 5km', '+ de 5km']
for j in range(tam):
    b_patch.append(mpatches.Patch(color=cores[j], label=labels[j]))
plt.legend(handles=b_patch)
plt.show();
fig = plt.figure(figsize=(12, 8))
plt.tight_layout()
ax = plt.subplot(211) 
ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
# Limits
plt.xlim(0,14)
plt.ylim(0,500)
# Color patterns
cm = plt.cm.get_cmap("rainbow")
xval = np.multiply(data.preco,1/1000000)
cor = cm(np.multiply(yval,5/7))
plt.scatter(xval, data.area, facecolor=cor, edgecolor='black')
plt.title (u'Precos vs area (scatered disperse vision)', fontsize=16, fontweight='bold', color='gray')
plt.ylabel(u'Área (m2)', fontsize=16, fontweight='bold', color='gray') 
# plt.xlabel(u'Dispersão em cores (R$1Mi)', fontsize=16, fontweight='bold')
# plt.text(-0.5, -55, "Fonte: sites de anúncios de imóveis | "  
#        "Autor: Roberto Barberá", fontsize=10, fontweight='bold')
#
# Add 2nd plot to the figure
ax1 = plt.subplot(212)
fig.add_subplot(ax1)
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
# Redefine axes limits
plt.xlim(0,14)
plt.ylim(0,500)
cm = plt.cm.get_cmap("viridis")
cor = cm(np.multiply(xval,1/20))
# Colors patterns
plt.scatter(np.multiply(data.preco,1/1000000), data.area, facecolor=cor)
plt.title (u'Precos vs area (scatered dense vision)', fontsize=16, fontweight='bold', color='gray')
plt.ylabel(u'Area (m2)', fontsize=16, fontweight='bold', color='gray') 
plt.xlabel(u'Precos (BRL 1Mi)', fontsize=16, fontweight='bold', color='gray')
plt.text(0, -100, "source: real state announces | Autor: Roberto Barberá", fontsize=10, fontweight='bold')
plt.show();
# Pr. M2 vs area >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PM2 vs area
#
fig = plt.figure(figsize=(12, 8))
plt.rc('axes', axisbelow=True)
ax = plt.subplot(211)
ax.yaxis.grid(color='lightgray', linestyle='dashed')
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left() 
#
plt.xlim(0,80000)
plt.ylim(0,500)
cm = plt.cm.get_cmap("rainbow")
cor = cm(np.multiply(xval,1/8))
#
plt.scatter(np.multiply(data.pm2,1), data.area, facecolor=cor, edgecolor='black')
plt.title (u'Pm2 vs area (disperse vision)', fontsize=16, fontweight='bold', color='gray')
plt.ylabel(u'Area (m2)', fontsize=16, fontweight='bold', color='gray') 
#
# Adds second plot to fig
ax1 = plt.subplot(212)
fig.add_subplot(ax1)
# ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
plt.xlim(0,80000)
plt.ylim(0,500)
plt.scatter(np.multiply(data.pm2,1), data.area, facecolor='black')
plt.title (u'Pm2 vs area (dense vision)', fontsize=16, fontweight='bold', color='gray')
plt.ylabel(u'Area (m2)', fontsize=16, fontweight='bold', color='gray') 
plt.xlabel(u'Precos (BRL 1.00)', fontsize=16, fontweight='bold', color='gray')
plt.text(0, -130, "source: real state announces | "  
         "Autor: Roberto Barberá", fontsize=10, fontweight='bold')
plt.show();
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Área/Distância vs Preco
# 
fig = plt.figure(figsize=(12, 8))
plt.rc('axes', axisbelow=True)
ax1 = plt.subplot(111)
ax1.yaxis.grid(color='lightgreen', linestyle='dashed')
cm = plt.cm.get_cmap("Greens")
fig.add_subplot(ax1)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom()  
ax1.get_yaxis().tick_left() 
# Redefine axes limits
plt.xlim(-0.5,14)
plt.ylim(-0.5,6.5)
# Color patterns
xval = np.multiply(data.preco,1/1000000)
yval = np.add(np.log1p(data.area), -np.log1p(np.multiply(data.distancia,1/10)))
cor = cm(np.multiply(xval,2/7))
plt.scatter(xval,yval, facecolor=cor, edgecolor='black')
# Titulos  e labels
plt.title (u'Area (weighted by inverse of distance) vs Preco', fontsize=16, fontweight='bold', color='gray')
plt.ylabel(u'Area/distance', fontsize=16, fontweight='bold', color='gray') 
plt.xlabel(u'Preços (BRL 1Mi)', fontsize=16, fontweight='bold', color='gray')
# Fonte, creditos e autoria
plt.text(-0.5, -1.5, "source: sites of real state advertises | "  
"Autor: Roberto Barberá", fontsize=10, fontweight='bold')
plt.show();
# Return the log(1+X) for every cell in the dataset
logdata = data.applymap(np.log1p)
corrmat = logdata.corr()
f, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(corrmat, vmax=.8, square=True, cmap="magma_r", linewidths=.5, center=0);
label = 'Correlation matrix of  variables'
plt.title (label, fontsize=14, fontweight='bold')
plt.show();
# preco correlation matrix
# 1st graph
fig = plt.figure(figsize=(16, 7))
plt.tight_layout()
ax = plt.subplot(121) 
fig.add_subplot(ax)
k = 10 # number of variables for heatmap (here all)
logdata = data.applymap(np.log1p)
corrmat = logdata.corr()
cols = corrmat.nlargest(k, 'preco')['preco'].index
cm = np.corrcoef(logdata[cols].values.T)
sns.set(font_scale=1.10)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', cmap="magma", annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
label = 'Preco correlation matrix'
plt.title (label, fontsize=14, fontweight='bold')
# 2nd graph
ax1 = plt.subplot(122) 
fig.add_subplot(ax1)
k = 10 # number of variables for heatmap (here all)
cols = corrmat.nlargest(k, 'pm2')['pm2'].index
cm = np.corrcoef(logdata[cols].values.T)
sns.set(font_scale=1.10)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', cmap="GnBu", annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
label = ' PM2 correlation matrix'
plt.title (label, fontsize=14, fontweight='bold')
plt.show();
# matplotlib scatterplots
sns.set()
logdata = data.applymap(np.log1p)
# unapply log to the columns 'quartos' and 'vagas'
logdata['quartos'] = data['quartos']
logdata['vagas'] = data['vagas']
logdata['area'] = data['area']
cols = ['preco', 'condominio', 'area', 'quartos', 'vagas']
sns.pairplot(logdata[cols], size = 2.5, hue='quartos')
plt.show();
# matplotlib scatterplots
sns.set()
logdata = data.applymap(np.log1p)
# unapply log to the column 'quartos' and 'vagas'
logdata['distancia'] = data['distancia']
logdata['quartos'] = data['quartos']
logdata['vagas'] = data['vagas']
logdata['area'] = data['area']
cols1 = ['pm2', 'preco', 'distancia', 'quartos']
sns.pairplot(logdata[cols1], size = 2.5, hue='quartos')
plt.show();
# histogram and normal probability plot
import scipy.stats as stats
from scipy.stats import norm
fig0 = plt.figure(figsize=(14, 5))        
ax2 = plt.subplot(121)  
fig0.add_subplot(ax2)
ax2.spines["top"].set_visible(False)  
ax2.spines["right"].set_visible(False)  
field = data['preco']
sns.distplot(field, fit=norm, color='darkgreen', kde_kws={"color": "r", "lw": 2})
# fig = plt.figure()
ax3 = plt.subplot(122) 
fig0.add_subplot(ax3)
res = stats.probplot(field, plot=ax3)
plt.show();
#histogram and normal probability plot
field = np.log2(data['preco'])
import scipy.stats as stats
from scipy.stats import norm
fig0 = plt.figure(figsize=(14, 5))        
ax2 = plt.subplot(121)  
fig0.add_subplot(ax2)
ax2.spines["top"].set_visible(False)  
ax2.spines["right"].set_visible(False)  
sns.distplot(field, fit=norm, color='xkcd:darkgreen', kde_kws={"color": "r", "lw": 2})
# fig = plt.figure()
ax3 = plt.subplot(122) 
fig0.add_subplot(ax3)
res = stats.probplot(field, plot=ax3)
plt.show();
# Bivariated plots
# pm2 x area
xval = data['pm2']
xvalog2 = np.log2(xval)
fig3 = plt.figure(figsize=(14, 5))
ax2 = plt.subplot(121) 
fig3.add_subplot(ax2)
ax2.spines["top"].set_visible(False)  
ax2.spines["right"].set_visible(False)  
ax2.get_xaxis().tick_bottom()  
ax2.get_yaxis().tick_left() 
# Redefine limites dos eixos
plt.xlim(11, 16)
# plt.ylim(0,500)
# cm = plt.cm.get_cmap("Greens")
# cor = cm(np.multiply(xval,1/1))
# Padroes de cores (usadas cores sólidas)
plt.scatter(xvalog2, data.area, facecolor='xkcd:blue')
plt.title (u'Pm2 vs area (dense vision)', fontsize=12, fontweight='bold', color='k')
plt.ylabel(u'Área (m2)', fontsize=12, fontweight='bold', color='k') 
# ax1.xaxis.set_label_coords(12, -55)
plt.xlabel(u'Precos of m$2$ (log2 BRL Mi)', fontsize=12, fontweight='bold', color='k')
plt.text(11, -100, "source: sites of real state advertises | Autor: Roberto Barberá", fontsize=10, fontweight='bold')
# preco(log2) x area
xval = data['preco']
xvalog2 = np.log2(xval)
ax1 = plt.subplot(122) 
fig3.add_subplot(ax1)
ax1.spines["top"].set_visible(False)  
ax1.spines["right"].set_visible(False)  
ax1.get_xaxis().tick_bottom() 
ax1.get_yaxis().tick_left() 
# Redefine limites dos eixos
plt.xlim(17, 25)
# plt.ylim(0,500)
# cm = plt.cm.get_cmap("Purples")
# cor = cm(np.multiply(xval,1/1))
# Padroes de cores (usadas cores sólidas)
plt.scatter(xvalog2, data.area, facecolor='xkcd:navy')
plt.title (u'Precos vs area (demse vision)', fontsize=12, fontweight='bold', color='k')
plt.ylabel(u'Área (m2)', fontsize=12, fontweight='bold', color='k') 
# ax1.xaxis.set_label_coords(12, -55)
plt.xlabel(u'Precos (log2 BRL Mi)', fontsize=12, fontweight='bold', color='k')
# plt.text(0, -80, "Fonte: sites de anúncios de imóveis | Autor: Roberto Barberá", fontsize=10, fontweight='bold')
plt.show();
# convert categorical variable into dummy
dataDum = pd.get_dummies(data)
total = data.isnull().sum().sort_values(ascending=False)
perc = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total, perc], axis=1, keys=['Total', ' Percentuals'])
missing.head(10)
# standardizing data
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
preco_sc = StandardScaler().fit_transform(data['preco'][:, np.newaxis]) ;
lim_inf = preco_sc[preco_sc[:,0].argsort()][:10]
lim_sup= preco_sc[preco_sc[:,0].argsort()][-10:]
print('Lower limit range of the distribution:');
print(lim_inf);
print('Higher limit range of the distribution:');
print(lim_sup);
import xgboost as xgb
model = xgb.XGBRegressor()
model.__dict__
dfpred = pd.read_csv("../input/prediction-samples/pred_saida.csv", delimiter=";", encoding="cp1252")
display(dfpred.head())