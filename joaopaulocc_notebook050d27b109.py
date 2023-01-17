# Bibliotecas do python

import numpy as np # para arrays mais eficientes e várias funções matemáticas

import matplotlib.pyplot as plt # para fazer gráficos

import ipywidgets as wdgts # Para fazer partes interativas
# Função que faz vários gráficos de máquina linear

def maquinaLinear(VB = 200, Fapl = 0, R = 0.1, B = 0.2, l = 0.1, m = 0.497, T = 500, dt = 0.1):

    '''Produz os gráficos de tensão induzida, corrente no laço, velocidade e Força induzida e Potência.

    VB - Tensão da bateria em Volts

    Fapl - Força aplicada em N. Positiva no sentido do movimento.

    R - Resistência do laço em Ohms.

    B - Densidade do campo magnético em Tesla.

    l - distância entre os trilhos, em metros.

    m - Massa, em kilogramas.

    T - Período da simulação, em segundos.

    dt - Passo da simulação, em segundos.'''

    fig = plt.figure(figsize=(16, 6), dpi=80)

    v0 = 0

    dt = 0.1

    T = 500

    index = 0

    t0 = 0

    t = np.arange(t0,T+t0+dt,dt)

    eind = []

    i = []

    Find = []

    v = [v0]

    Pel = []

    Pmec = []

    while index*dt <= T:

        eind.append(v[index]*B*l)

        i.append((VB-eind[index])/R)

        Find.append(i[index]*l*B)

        Ft = Find[index]+Fapl

        a = Ft/m

        Pel.append(eind[index]*i[index])

        Pmec.append(Find[index]*v[index])

        v.append(v[index] + a * dt)

        index = index + 1

    v.pop()

    ax1 = fig.add_subplot(2,2,1)

    ax1b = ax1.twinx()

    ax2 = fig.add_subplot(2,2,3)

    ax2b = ax2.twinx()

    ax3 = fig.add_subplot(1,2,2)

    ax1.plot (t, eind, color="blue", lw=2)

    ax1b.plot (t, i, color="red", lw=2)

    ax1.set_ylabel("Tensão induzida [V] azul")

    ax1b.set_ylabel("Corrente [A] vermelho")

    ax2.plot (t, v, color="blue", lw=2)

    ax2b.plot (t, Find, color="red", lw=2)

    ax2.set_ylabel("Velocidade [m/s] azul")

    ax2b.set_ylabel("Força [N] vermelho")

    ax2.set_xlabel("Tempo [s]")

    ax3.plot(t,Pel, color="blue", lw=2)

    ax3.plot(t,Pmec, color="red", lw=2)

    ax3.set_ylabel("Potência [W]")

    ax3.set_xlabel("Tempo [s]")
# Gráfico com Sliders para vários parâmetros da máquina linear

VBw=wdgts.FloatSlider(min=0,max=400,value=200,description="Tensão da bateria")

Faplw=wdgts.FloatSlider(min=-1.0,max=1.0,value=0.0,description="Força aplicada")

Rw=wdgts.FloatSlider(min=0.1,max=10.0,value=0.1,description="Resistência")

Bw=wdgts.FloatSlider(min=0.1,max=50.0,value=0.2,description="Campo Magnético")

lw=wdgts.FloatSlider(min=0.01,max=1.0,value=0.1,description="Comprimento")

mw=wdgts.FloatSlider(min=0.1,max=10,value=0.497,description="Massa")

leftBox = wdgts.VBox([VBw,Faplw,Rw,Bw,lw,mw])

#leftBox

plots = wdgts.interactive(maquinaLinear,

              VB=VBw,

              Fapl=Faplw,

              R=Rw,

              B=Bw,

              l=lw,

              m=mw)

rightBox = wdgts.HBox([plots])

rightBox
# Função que faz vários gráficos até o comprimento de trilho dist

def maquinaLinear2(VB = 200, Fapl = 0, R = 0.1, B = 0.2, l = 0.1, m = 0.497,dist=2, dt = 0.001):

    '''Produz os gráficos de tensão induzida, corrente no laço, velocidade e Força induzida e Potência.

    VB - Tensão da bateria em Volts

    Fapl - Força aplicada em N. Positiva no sentido do movimento.

    R - Resistência do laço em Ohms.

    B - Densidade do campo magnético em Tesla.

    l - distância entre os trilhos, em metros.

    m - Massa, em kilogramas.

    dist - Comprimento do trilho, em metros.

    dt - Passo da simulação, em segundos.'''

    fig = plt.figure(figsize=(16, 6), dpi=80)

    v0 = 0

    dt = 0.001

    index = 0

    t0 = 0

    t=[t0]

#     t = np.arange(t0,T+t0+dt,dt)

    eind = []

    i = []

    Find = []

    v = [v0]

    x0 = 0

    x = [x0]

    Pel = []

    Pmec = []

    #while index*dt <= T:

    while x[-1] <= dist:

        eind.append(v[index]*B*l)

        i.append((VB-eind[index])/R)

        Find.append(i[index]*l*B)

        Ft = Find[index]+Fapl

        a = Ft/m

        Pel.append(eind[index]*i[index])

        Pmec.append(Find[index]*v[index])

        t.append(t[index]+dt)

        v.append(v[index] + a * dt)

        x.append(x[index] + v[index] * dt)

        index = index + 1

#         print(x[-1])

    t.pop()

    v.pop()

    x.pop()

    ax1 = fig.add_subplot(2,2,1)

    ax1b = ax1.twinx()

    ax2 = fig.add_subplot(2,2,3)

    ax2b = ax2.twinx()

    ax3 = fig.add_subplot(1,2,2)

    ax1.plot (t, eind, color="blue", lw=2)

    ax1b.plot (t, i, color="red", lw=2)

    ax1.set_ylabel("Tensão induzida [V] azul")

    ax1b.set_ylabel("Corrente [A] vermelho")

    ax2.plot (t, v, color="blue", lw=2)

    ax2b.plot (t, Find, color="red", lw=2)

    ax2.set_ylabel("Velocidade [m/s] azul")

    ax2b.set_ylabel("Força [N] vermelho")

    ax2.set_xlabel("Tempo [s]")

    ax3.plot(t,Pel, color="blue", lw=2)

    ax3.plot(t,Pmec, color="red", lw=2)

    ax3.set_ylabel("Potência [W]")

    ax3.set_xlabel("Tempo [s]")

    return {"t":t,"v":v,"Pel":Pel}
# Gráfico com Sliders para vários parâmetros da máquina linear

VBw=wdgts.FloatSlider(min=0,max=400,value=200,description="Tensão da bateria")

Faplw=wdgts.FloatSlider(min=-1.0,max=1.0,value=0.0,description="Força aplicada")

Rw=wdgts.FloatSlider(min=0.1,max=10.0,value=0.1,description="Resistência")

Bw=wdgts.FloatSlider(min=0.1,max=50.0,value=0.2,description="Campo Magnético")

lw=wdgts.FloatSlider(min=0.01,max=1.0,value=0.1,description="Comprimento")

mw=wdgts.FloatSlider(min=0.1,max=10,value=0.497,description="Massa")

leftBox = wdgts.VBox([VBw,Faplw,Rw,Bw,lw,mw])

#leftBox

plots = wdgts.interactive(maquinaLinear2,

              VB=VBw,

              Fapl=Faplw,

              R=Rw,

              B=Bw,

              l=lw,

              m=mw)

rightBox = wdgts.HBox([plots])

rightBox
dados = maquinaLinear2()

dados["v"][-1] #Ultimo valor de velocidade