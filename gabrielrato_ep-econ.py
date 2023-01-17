import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

from plotly.graph_objs import Scatter, Layout, Marker
import plotly.graph_objs as go
import random


def oferta_demanda():
    "Calcula a Oferta e demanda perante mudança de gostos"

    # construir parametrizacao random e printar ela descrita
    # varia de 0 a 10
    gostos = []
    for i in range(3):
        gostos.append(random.randint(0,10))
    gostos = sorted(gostos)
    expectativa = 6

    traces = []
    for b in [0.5, 8]:
        for index, gosto in enumerate(gostos):
            delta = (gosto + expectativa)/20.0
            if b == 8:
                delta = -1*delta
            axis_x = []
            axis_y = []
            for x in range(10):
                axis_x.append(x)
                axis_y.append(b + float(delta*x))

            trace = go.Scatter(
                x=axis_x,
                y=axis_y,
                name = 'Oferta_'+str(index) if b==8 else 'Demanda_'+str(index)
            )
            traces.append(trace)

    fig = {
        "data": [trace for trace in traces],
        "layout": Layout(title="CURVA OFERTA & DEMANDA" )
    }


    iplot(fig)

oferta_demanda()



def elasticidade():
    "Calcula a elasticidade de um determinado produto"

    elasticidade = ['simples', 'perfeita_inelastica', 'perfeita_elastica']
    for ele in elasticidade:
        axis_x = []
        traces = []
        axis_y = []

        # calc a simple demand curve
        for x in range(10):
            if ele == 'perfeita_inelastica':
                axis_x.append(7)
            elif ele == 'perfeita_elastica':
                axis_x.append(x*1000)
            else:
                axis_x.append(x)
            axis_y.append(8 - float(2*x))

        trace = go.Scatter(
            x=axis_x,
            y=axis_y,
            name='demanda'
        )
        traces.append(trace)

        # calc a elasticity curve
        new_axis_y = []
        new_axis_x = []
        for x in range(9):
            if x % 2 == 1:
                continue
            new_axis_x.append(x)
            elasticity = float(((axis_x[x] - axis_x[x+1])/ (axis_x[x]+axis_x[x+1])/2.0)) \
                / float(((axis_y[x] - axis_y[x+1])/ (axis_y[x] + axis_x[x+1]/2.0)))

            new_axis_y.append(elasticity)

        trace = go.Scatter(
            x=new_axis_x,
            y=new_axis_y,
            name='elasticidade'

        )
        traces.append(trace)
        fig = {
            "data": [trace for trace in traces],
            "layout": Layout(title="CURVA "+ele.upper() )
        }


        iplot(fig)
elasticidade()

def pib():
    "Calcula o pib real e nominal ao longo de alguns anos"
    anos_dinheiro = {
        2000 : 100,
        2001 : 110,
        2002 : 90,
        2003 : 105,
        2004 : 120,
    }
    anos_preco = {
        2000 : 0.55,
        2001 : 0.6,
        2002 : 0.7,
        2003 : 0.45,
        2004 : 0.8,
    }
    traces = []

    for tipo in ['preco_unit (* 100)', 'nominal', 'real']:
        axis_y = []
        axis_x = []
        for ano in anos_dinheiro:
            if tipo == 'nominal':
                axis_x.append(ano)
                axis_y.append(anos_dinheiro[ano])
            if tipo == 'preco_unit (* 100)':
                axis_x.append(ano)
                axis_y.append(anos_preco[ano]*100)
            if tipo =='real':
                axis_x.append(ano)
                axis_y.append(anos_dinheiro[ano]/anos_preco[ano] \
                    * anos_preco[2000])

        trace = go.Scatter(
            x=axis_x,
            y=axis_y,
            name = tipo
        )
        traces.append(trace)

    fig = {
        "data": [trace for trace in traces],
        "layout": Layout(title="CURVA PIB" )
    }

    iplot(fig)

pib()
def inflacao():
    "Calcula inflacao para uma nacao simples"
    anos_dinheiro = {
        2000 : [100, 100],
        2001 : [150, 180],
        2002 : [155, 182],
        2003 : [165, 190],
        2004 : [166, 193],
        2005 : [150, 180],
        2006 : [162, 184],
        2007 : [155, 193],

    }
    percent = [0.6, 0.4]
    traces = []

    axis_y = []
    axis_x = []
    base = anos_dinheiro[2000][0] * percent[0] + anos_dinheiro[2000][1] * percent[1]
    for ano in anos_dinheiro:
        axis_x.append(ano)
        tot = anos_dinheiro[ano][0]*percent[0] + anos_dinheiro[ano][1]*percent[1]
        axis_y.append((tot - base))
    traces = []
    trace = go.Scatter(
        x=axis_x,
        y=axis_y,
        name = 'Inflacao em %'
    )
    traces.append(trace)
    axis_y1 = []
    axis_y2 = []
    for ano in anos_dinheiro:
        axis_y1.append(anos_dinheiro[ano][0])
        axis_y2.append(anos_dinheiro[ano][1])

    for index, y in enumerate([axis_y1, axis_y2]):
        trace = go.Scatter(
            x=axis_x,
            y=y,
            name = 'Produto_'+str(index)
        )
        traces.append(trace)


    fig = {
        "data": [trace for trace in traces],
        "layout": Layout(title="CURVA INFLACAO" )
    }

    iplot(fig)

inflacao()
import random

# Tamanho da populacao e media 
N  = 5000 
MU = 100. 

population = [random.gauss(mu=MU, sigma=MU/5) for actor in range(N)]

def gini(y):
    " Calculo do indice gini para populacao y"
    y = sorted(y)
    n = len(y)
    numer = 2 * sum((i+1) * y[i] for i in range(n))
    denom = n * sum(y)
    return (numer / denom) - (n + 1) / n

%matplotlib inline
import matplotlib.pyplot as plt

def hist(population, label='pop', **kwargs):
    "Histograma para uma populacao"
    label = label + ': G=' + str(round(gini(population), 2))
    h = plt.hist(list(population), bins=30, alpha=0.5, label=label, **kwargs)
    plt.xlabel('renda'); plt.ylabel('qtd'); plt.grid(True)
    plt.legend()
hist(population)

def random_split(A, B):
    " Pega todo o dinheiro, insere em um pote e divide randomicamente entre os atores "
    pot = A + B
    share = random.uniform(0, pot)
    return share, pot - share
random_split(100, 100)

def anyone(N): return random.sample(range(N), 2)
def simulate(population, T, transaction=random_split, interaction=anyone):
    " Roda uma simulacao sobre uma populacao para T transacoes "

    population = population.copy()
    yield population
    for t in range(1, T + 1):
        i, j = interaction(len(population))
        population[i], population[j] = transaction(population[i], population[j]) 
        yield population
for pop in simulate([100] * 4, 8):
    print(pop)
import statistics

def show(population, k=40, percentiles=(1, 10, 50, 90, 99), **kwargs):
    "Run a simulation for k*N steps, printing statistics and displaying a plot and histogram."
    N = len(population)
    start = list(population)
    # Sort results so that percentiles work
    results = [(t, sorted(pop)) 
               for (t, pop) in enumerate(simulate(population, k * N, **kwargs))
               if t % (N / 10) == 0]
    times = [t for (t, pop) in results]
    
    # Printout:
    print('   t    Gini stdev' + (' {:3d}%' * len(percentiles)).format(*percentiles))
    print('------- ---- -----' + ' ----' * len(percentiles))
    fmt = '{:7,d} {:.2f} {:5.1f}' + ' {:4.0f}' * len(percentiles)
    for (t, pop) in results:
        if t % (4 * N) == 0:
            data = [percent(pct, pop) for pct in percentiles]
            print(fmt.format(t, gini(pop), statistics.stdev(pop), *data))
            
    # Plot:
    plt.hold(True); plt.xlabel('renda'); plt.ylabel('iter'); plt.grid(True)
    for pct in percentiles:
        line = [percent(pct, pop) for (t, pop) in results]
        plt.plot(line, times)
    plt.show()
    
    # Histogram:
    R = (min(pop+start), max(pop+start))
    hist(start, 'inicio', range=R)
    hist(pop, 'fim', range=R)
    
    return pop
                
def percent(pct, items):
    "The item that is pct percent through the sorted list of items."
    return items[min(len(items)-1, len(items) * pct // 100)]
pop = show(population)

"Pega os 1% mais ricos e cobra um tributo de 25%, paga esse imposto e retorna para a populacao"
ini_pop = pop
for ite in range(100):
    
    richest =  sorted(pop)[4950:]
    tax = 0
    new_richest = []
    for actor in sorted(richest):
        new_richest.append(actor * 0.75)
        tax += actor * 0.25

    pop = pop[:4950] + new_richest

    new_pop = []
    for actor in pop:
        actor += tax/len(pop)
        new_pop.append(actor)
    pop = new_pop

R = (min(ini_pop+pop), max(ini_pop+pop))
hist(ini_pop, 'inicio', range=R)
hist(pop, 'fim', range=R)
