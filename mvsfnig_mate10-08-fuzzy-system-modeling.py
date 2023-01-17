import numpy as np               # linear algebra - vectors and matrices
import matplotlib.pyplot as plt  # visualization
import matplotlib.cm as cm       # visualization
import seaborn as sns            # visualization
%matplotlib inline
sns.set()
class fuzzy:
    
    def __init__(self,size_set,conjunto_x=None, conjunto_y=None):
        
        self.x = np.zeros(size_set)
        self.y = np.zeros(size_set)
        self.set = np.zeros((size_set,2))
        self.size = size_set
        
        if conjunto_x:
            self.set_x(conjunto_x)
        
        if conjunto_y:
            self.set_y(conjunto_y)
            
        if conjunto_x and conjunto_y:
            self.merge()
        
    def set_x(self, conjunto):
        """
        seta o conjunto passado ao conjunto x - dominio
        """
        for i in range(self.x.shape[0]):
            self.x[i] = conjunto[i]
            
    def set_y(self, conjunto):
        """
        seta o conjunto passado ao conjunto y - pertinencias
        """
        for i in range(self.y.shape[0]):
            self.y[i] = conjunto[i]
            
    
    def merge(self,verbose=False):
        """
        junta o conjunto x com y em um unico array
        """
        for i in range(self.x.shape[0]):
            self.set[i][0] = self.x[i]
            self.set[i][1] = self.y[i]
            
        if verbose:
            print(self.set)
        
    def get_y(self, x):
        """
        obtem o valor da pertinencia do valor x passado
        """
        for i in range(self.x.shape[0]):
            if x == self.x[i]:
                return self.y[i]
        return 0
    
    def view_set(self,):
        print(self.set)
        

def triangular(a,m,b,x=None, p=None, max_degree=None):
            """
            define pertinence y of the triangular function from x values
            a - primeiro valor conjunto
            m - valor mediano do conjunto
            b - último valor do conjunto
            x - valor a ser calculado
            """
            # IF x
            if x:
                if x <= a or x >= b:
                    return 0

                if x == m and max_degree != None:
                    return max_degree

                if x > a and x <= m:
                    return ((x-a)/(m-a))

                if x > m and x < b:
                    return ((b-x)/(b-m))
            else:
                x_after = (p * (m-a))+a
                x_before = (((p * (b-m))-b) * -1)
                return x_after, x_before
            
def trapezoidal(a,m,n,b,x,max_degree = None):
            """
            define pertinence y of the trapezoidal function from x values
            a - primeiro valor conjunto
            m - primeiro valor com maximo valor de pertinência
            n - primeiro valor com maximo valor de pertinência
            b - último valor do conjunto
            x - valor a ser calculado
            """
                
            if max_degree is None:
                max_degree = 1

            if x <= a or x >= b:
                return 0
            
            if x >= m and x <= n:
                return max_degree

            if x > a and x < m:
                return (x-a)/(m-a)
    
            if x > n and x < b:
                return (b-x)/(b-n)
def plot_sets(conjuntos, labels, title=None, position=None):
    """
     - função para plotagem dos conjuntos 2d
     -- plota os conjuntos em um mesmo plano
    """
    
    #color = ['--bo','--go', '--ro', '--yo', '--po']
    facecolors = [cm.jet(x) for x in np.random.rand(20)]
    
    fig, ax = plt.subplots(figsize=(8,5))

    indice = 0
    
    for i in conjuntos:
        
        if 'area' in labels[indice]:
            labels[indice] = labels[indice].split('area')[0]
            plt.fill_between(i[:,0], i[:,1], alpha=0.4)  
            ax.plot(i[:,0], i[:,1], label=labels[indice], linewidth=4)
        else:
            ax.plot(i[:,0], i[:,1], label=labels[indice])
            
        indice += 1 
            
    if position:
        legend = ax.legend(loc=position, shadow=True, fontsize='x-large')
    else:
        legend = ax.legend(loc=9, shadow=True, fontsize='medium',
                           bbox_to_anchor=(0.5, -0.1), ncol=len(labels))

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#f2f2f2')
    plt.title(title)
    plt.grid(True)
# visualização dos conjuntos
def plot_tensor(tensor, labels=None,  subtitles='Nan', position=None, dim=[1,1], size=(15,4)):
    """
     - função para plotagem dos conjuntos 2d
     tensor.....: matriz de gráficos para plotagem
     labels ....: objetos descritos de cada gráfico
     subtitles..: subtitle of each graphs
     position ..: local legend
     dim........: list 2d with firt position wor, and tow col
    """
    
    facecolors = [cm.jet(x) for x in np.random.rand(20)]
    
    fig, ax = plt.subplots(dim[0], dim[1], figsize=size)
    fig.subplots_adjust(top=0.85)

    indice = 0
    
    for row in range(dim[0]): # linha da matriz de graficos
        
        for i in range(dim[1]): # coluna da matriz de gráficos

            if subtitles[0][i] != 'Nan':
                ax[i].set_title(label=subtitles[0][i],loc='center')

                
            for j in range(tensor[i].shape[0]): # for que ploa os conjuntos
                
                if 'area' in labels[i][j]:
                    labels[i][j] = labels[i][j].split('area')[0]
                    ax[i].fill_between(tensor[i][j][:,0], tensor[i][j][:,1], alpha=0.4) 
                    ax[i].plot(tensor[i][j][:,0], tensor[i][j][:,1], label=labels[i][j], linewidth=4)
                else:
                    ax[i].plot(tensor[i][j][:,0], tensor[i][j][:,1], label=labels[i][j])
                
                        
            if position:
                legend = ax[i].legend(loc=position, shadow=True, fontsize='x-large')
            else:
                legend = ax[i].legend(loc=9, shadow=True, fontsize='medium',  
                                      bbox_to_anchor=(0.5, -0.15), ncol=len(labels))
            indice += 1

    legend.get_frame().set_facecolor('#f2f2f2')
    plt.grid(True)
# Variação da vendas V - definindo os conjuntos fuzzy
vendas_dim = fuzzy(size_set=4,conjunto_x=[-100,-100,-50,0],conjunto_y=[0,1,1,0])

vendas_est = fuzzy(size_set=3,conjunto_x=[-50,0,50],conjunto_y=[0,1,0])

vendas_aum = fuzzy(size_set=4,conjunto_x=[0,50,100,100],conjunto_y=[0,1,1,0])
s_baixa = fuzzy(size_set=3,conjunto_x=[0,0,50],conjunto_y=[0,1,0]) 

s_media = fuzzy(size_set=3,conjunto_x=[0,50,100],conjunto_y=[0,1,0]) 

s_alta = fuzzy(size_set=3,conjunto_x=[50,100,100],conjunto_y=[0,1,0]) 
i_ruim = fuzzy(size_set=3,conjunto_x=[0,0,50],conjunto_y=[0,1,0]) 

i_medio = fuzzy(size_set=3,conjunto_x=[0,50,100],conjunto_y=[0,1,0])

i_bom = fuzzy(size_set=3,conjunto_x=[50,100,100],conjunto_y=[0,1,0])
# 
tensor = np.array([[vendas_dim.set,vendas_est.set,vendas_aum.set],
                       [s_baixa.set,s_media.set,s_alta.set],
                        [i_ruim.set,i_medio.set,i_bom.set]])

subtitles = np.array([['Variação das Vendas', 'Sobrecarga de Serviços',
                       'Nível de informatização']])

labels = np.array([['Diminuindo', 'Estável', 'Aumentando'],
                  ['baixa', 'média', 'alta'],
                  ['Ruim', 'Médio', 'Bom']])

plot_tensor(tensor, labels=labels, subtitles=subtitles,
            position=None, dim=[1,tensor.shape[1]])
r_ruim = fuzzy(size_set=3,conjunto_x=[0,0,50],conjunto_y=[0,1,0]) 

r_medio = fuzzy(size_set=3,conjunto_x=[0,50,100],conjunto_y=[0,1,0])

r_bom = fuzzy(size_set=3,conjunto_x=[50,100,100],conjunto_y=[0,1,0]) 

plot_sets(np.array([i_ruim.set,i_medio.set,i_bom.set]),
          ['Leve', 'Média', 'Forte'],
          'Recomendação de Investimento (R)') # visualizando os conjuntos

tensor = np.array([np.array([vendas_dim.set,vendas_est.set,vendas_aum.set,
                             np.array([[55,0], [55,1]], dtype='float32')]),
                       np.array([s_baixa.set,s_media.set,s_alta.set,
                                 np.array([[60,0], [60,1]], dtype='float32')]),
                        np.array([i_ruim.set,i_medio.set,i_bom.set,
                                  np.array([[85,0], [85,1]], dtype='float32')])])

subtitles = np.array([['Variação das Vendas', 'Sobrecarga de Serviços',
                       'Nível de informatização']])

labels = np.array([['Diminuindo', 'Estável', 'Aumentando','Entrada'],
                  ['baixa', 'média', 'alta', 'Entrada'],
                  ['Ruim', 'Médio', 'Bom', 'Entrada']])
# aumentando
plot_tensor(tensor, labels=labels, subtitles=subtitles, dim=[1,3])
in_vendas_aum = trapezoidal(a=0,m=50,n=100,b=100,x=55)
in_servi_alta = triangular(a=50,m=100,b=100,x=85)
in_info_bom = triangular(a=50,m=100,b=100,x=60)

in_servi_media = triangular(a=0,m=50,b=100,x=60)
in_servi_baixa = 0
in_info_ruim = 0
# obtendo o valor da regra
R1_mandani = min(in_vendas_aum,in_servi_alta,in_info_bom)

# encontrando as novas coordenadas de x que valor da regra 1 define
x_cob_inv = triangular(a=i_bom.x[0],m=i_bom.x[1],b=i_bom.x[2],p=R1_mandani)

# corte do Mandani
corte_mandani = np.array([[0,R1_mandani],[100,R1_mandani]]) 

# Área de corte do Mandani
r1_area_mandani = np.array([[i_bom.x[0],i_bom.y[0]],[x_cob_inv[0],R1_mandani],
                            [x_cob_inv[1],R1_mandani],[i_bom.x[2],i_bom.y[2]]])

# calculando o valor das pertinência com Larsen
y_larsen = i_bom.y * R1_mandani

# criando a área ṕara o gráfico
r1_larsen = np.array([[i_bom.x[0],y_larsen[0]],[i_bom.x[1],
                     y_larsen[1]],[i_bom.x[2],y_larsen[2]]])

# Matriz multidimensional
tensor = np.array([np.array([i_ruim.set, i_medio.set, i_bom.set,
                             corte_mandani, r1_area_mandani]),
                    np.array([i_ruim.set, i_medio.set, i_bom.set,
                              r1_larsen])])

subtitles = np.array([[' Recomendação de Investimento (R) - Regra 1 Mandani',
                       'Recomendação de Investimento (R) - Regra 1 Larsen']])

labels = np.array([['Leve', 'Média', 'Forte', ''+str(R1_mandani), 'Mandani area'],
                  ['Leve', 'Média', 'Forte', 'Larsen area']])

plot_tensor(tensor, labels=labels, subtitles=subtitles, dim=[1,2], size=(15,5))
R2_mandani = min(in_vendas_aum,in_servi_media,in_info_bom)

# econtrando as novas coordenadas de x que valor da regra 1 define
x_cob_inv = triangular(a=i_medio.x[0],m=i_medio.x[1],b=i_medio.x[2],p=R2_mandani)

# corte do Mandani
corte_mandani = np.array([[0,R2_mandani],[100,R2_mandani]])

# Área de corte do Mandani
r2_area_mandani = np.array([[i_medio.x[0],i_medio.y[0]],[x_cob_inv[0],R2_mandani],
                            [x_cob_inv[1],R2_mandani],[i_medio.x[2],i_medio.y[2]]])

# calculando o valor das pertinência com Larsen
y2_larsen = i_medio.y * R2_mandani

# criando a área ṕara o gráfico
r2_larsen = np.array([[i_medio.x[0],y2_larsen[0]],
                      [i_medio.x[1],y2_larsen[1]],
                      [i_medio.x[2],y2_larsen[2]]])

# Matriz multidimensional
tensor = np.array([
                    np.array([i_ruim.set, i_medio.set, i_bom.set,
                              corte_mandani, r2_area_mandani]),
                    np.array([i_ruim.set, i_medio.set, i_bom.set, r2_larsen])
                ])

subtitles = np.array([[' Recomendação de Investimento (R) - Regra 2 Mandani',
                       'Recomendação de Investimento (R) - Regra 2 Larsen']])

labels = np.array([['Leve', 'Média', 'Forte', ''+str(R2_mandani), 'Mandani area'],
                  ['Leve', 'Média', 'Forte', 'Larsen area']])

plot_tensor(tensor, labels=labels, subtitles=subtitles, dim=[1,2], size=(15,5))

R3_mandani = min(in_vendas_aum, in_servi_baixa, in_info_bom)

# econtrando as novas coordenadas de x que valor da regra 1 define
x_cob_i = triangular(a=i_ruim.x[0],m=i_ruim.x[1],b=i_ruim.x[2],p=R3_mandani)

# -------------------------------------------------------------------------------------------

# corte do Mandani
corte_mandani = np.array([[0,R3_mandani],[100,R3_mandani]])

# Área de corte do Mandani
r3_area_mandani = np.array([[i_ruim.x[0],i_ruim.y[0]],[x_cob_i[0],R3_mandani],
                            [x_cob_i[1],R3_mandani],[i_ruim.x[2],i_ruim.y[2]]])

# calculando o valor das pertinência com Larsen
y3_larsen = i_ruim.y * R3_mandani

# criando a área ṕara o gráfico
r3_larsen = np.array([[i_ruim.x[0],y3_larsen[0]],[i_ruim.x[1],y3_larsen[1]],
                      [i_ruim.x[2],y3_larsen[2]]])

# Matriz multidimensional
tensor = np.array([
                    np.array([i_ruim.set, i_medio.set, i_bom.set,
                              corte_mandani, r3_area_mandani]),
                    np.array([i_ruim.set, i_medio.set, i_bom.set,
                              r3_larsen])])

subtitles = np.array([[' Recomendação de Investimento (R) - Regra 3 Mandani',
                       'Recomendação de Investimento (R) - Regra 3 Larsen']])

labels = np.array([['Leve', 'Média', 'Forte', ''+str(R3_mandani), 'Mandani area'],
                  ['Leve', 'Média', 'Forte', 'Larsen area']])

plot_tensor(tensor, labels=labels, subtitles=subtitles, dim=[1,2], size=(15,5))
R4_mandani = min(in_vendas_aum, in_servi_media, in_info_ruim)

# encontrando as novas coordenadas de x que valor da regra 1 define
x_cob_inv = triangular(a=i_bom.x[0],m=i_bom.x[1],b=i_bom.x[2],p=R4_mandani)
# -------------------------------------------------------------------------
# corte do Mandani
corte_mandani = np.array([[0,R4_mandani],[100,R4_mandani]])

# Área de corte do Mandani
r4_area_mandani = np.array([[i_bom.x[0],i_bom.y[0]],[x_cob_inv[0],R4_mandani],
                            [x_cob_inv[1],R4_mandani],[i_bom.x[2],i_bom.y[2]]])

# calculando o valor das pertinência com Larsen
y4_larsen = i_ruim.y * R4_mandani

# criando a área ṕara o gráfico
r4_larsen = np.array([[i_bom.x[0],y4_larsen[0]],[i_bom.x[1],
                    y4_larsen[1]],[i_bom.x[2],y4_larsen[2]]])

# Matriz multidimensional
tensor = np.array([
                    np.array([i_ruim.set, i_medio.set, i_bom.set,
                              corte_mandani, r4_area_mandani]),
                    np.array([i_ruim.set, i_medio.set, i_bom.set,
                              r4_larsen])])

subtitles = np.array([[' Recomendação de Investimento (R) - Regra 4 Mandani',
                       'Recomendação de Investimento (R) - Regra 4 Larsen']])

labels = np.array([['Leve', 'Média', 'Forte', ''+str(R3_mandani), 'Mandani area'],
                  ['Leve', 'Média', 'Forte', 'Larsen area']])

plot_tensor(tensor, labels=labels, subtitles=subtitles, dim=[1,2], size=(15,5))
# Mandani
aggre_mandani = np.zeros(r1_area_mandani.shape)

pontos = [0,10,100,100]

for i in range(r1_area_mandani.shape[0]):
    aggre_mandani[i][1] = max(r1_area_mandani[i][1],
                        r2_area_mandani[i][1],
                        r3_area_mandani[i][1],
                        r4_area_mandani[i][1])
    aggre_mandani[i][0] = pontos[i] 
# Larsen
aggre_larsen = np.zeros(r2_larsen.shape)

pontos = [0,50,100]

for i in range(r2_larsen.shape[0]):
    aggre_larsen[i][1] = max(r1_larsen[i][1],
                        r2_larsen[i][1],
                        r3_larsen[i][1],
                        r4_larsen[i][1])
    aggre_larsen[i][0] = pontos[i] 
# Matriz multidimensional
tensor = np.array([[i_ruim.set, i_medio.set, i_bom.set, aggre_mandani],
                   [i_ruim.set, i_medio.set, i_bom.set, aggre_larsen]])


subtitles = np.array([[' Recomendação de Investimento (R) - Agregação Mandani',
                       'Recomendação de Investimento (R) - Agregação Larsen']])

labels = np.array([['Leve', 'Média', 'Forte', 'Mandani area'],
                  ['Leve', 'Média', 'Forte', 'Larsen area']]) # 'Larsen area'

plot_tensor(tensor, labels=labels, subtitles=subtitles, dim=[1,2], size=(15,5))
