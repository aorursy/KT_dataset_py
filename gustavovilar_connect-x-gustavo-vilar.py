#Precisamos fazer o download do ambiente de Connect X:
# Vamos precisar de um kaggle-enviroments customizado para a avaliação.
!pip install git+https://github.com/matheusgmaia/kaggle-environments
    
#Criar ambiente
from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)
env.render()
# Exemplo de agente. Esse agente escolhe de maneira aleatória uma coluna que não esteja completa
import random

def my_agent(obs, cfg): #recebe o estado atual do jogo e a configuração do jogo
    coluna = random.choice([c for c in range(cfg.columns) if obs.board[c] == 0])
    return coluna
env.reset()
env.run([my_agent, "random"]) #Agente definido em my_agent versus angente randômico.
env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = my_agent(observation, env.configuration)
    print("Ação do seu agente: Coluna", my_action+1)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
# "None" represents which agent you'll manually play as (first or second player).
env.play([None, my_agent], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
import numpy as np
import random
import bisect
fit = False
# conta o numero de 1 e 2 num raio de 3 casas na horizontal vertical e diagonais
def count(board, r, col):
    resp     = {'h':[0,0], 'v':[0,0], 'd':[0,0], 'f':0}
    counter  = {'h':[0,0], 'v':[0,0], 'd':[[0, 0], [0, 0]]}
    counter2 = {'h':[0,0], 'v':[0,0], 'd':[[0, 0], [0, 0]]}
    counter3 = {'h':[0,0], 'v':[0,0], 'd':[[0, 0], [0, 0]]}
  
    for i in range(-3, 4):
        if i==0:
            # se houver possibilidade de ganhar ou perder nessa jogada essa jogada deve ser feita
            if counter['h'][0]==3 or counter['h'][1]==3 or counter['v'][0]==3 or counter['v'][1]==3 or counter['d'][0][0]==3 or counter['d'][0][1]==3 or counter['d'][1][0]==3 or counter['d'][1][1]==3:
                resp['f'] = 1
                break
            counter  = {'h':[0,0], 'v':[0,0], 'd':[[0, 0], [0, 0]]}
            continue
            
        # conta na vertical
        if r+i>=0 and r+i<6:
            if board[r+i][col]==1:
                resp['v'][0]     += 1
                counter['v'][0]  += 1
                counter2['v'][0] += 1
                counter3['v'][0] += 1
            elif board[r+i][col]==2:
                resp['v'][1]     += 1
                counter['v'][1]  += 1
                counter2['v'][1] += 1
                counter3['v'][1] += 1
        # conta na horizontal
        if col+i>=0 and col+i<7:
            if board[r][col+i]==1:
                resp['h'][0]     += 1
                counter['h'][0]  += 1
                counter2['h'][0] += 1
                counter3['h'][0] += 1
            elif board[r][col+i]==2:
                resp['h'][1]     += 1
                counter['h'][1]  += 1
                counter2['h'][1] += 1
                counter3['h'][1] += 1
        # conta na diagonal principal
        if r+i>=0 and r+i<6 and col+i>=0 and col+i<7:
            if board[r+i][col+i]==1:
                resp['d'][0]        += 1
                counter['d'][0][0]  += 1
                counter2['d'][0][0] += 1
                counter3['d'][0][0] += 1
            elif board[r+i][col+i]==2:
                resp['d'][1]        += 1
                counter['d'][0][1]  += 1
                counter2['d'][0][1] += 1
                counter3['d'][0][1] += 1
        # conta na diagonal secundaria
        if r+i>=0 and r+i<6 and col-i>=0 and col-i<7:
            if board[r+i][col-i]==1:
                resp['d'][0]        += 1
                counter['d'][1][0]  += 1
                counter2['d'][1][0] += 1
                counter3['d'][1][0] += 1
            elif board[r+i][col-i]==2:
                resp['d'][1]        += 1
                counter['d'][1][1]  += 1
                counter2['d'][1][1] += 1
                counter3['d'][1][1] += 1
                
        if i==3 and (counter['h'][0]==3 or counter['h'][1]==3 or counter['v'][0]==3 or counter['v'][1]==3 or counter['d'][0][0]==3 or counter['d'][0][1]==3 or counter['d'][1][0]==3 or counter['d'][1][1]==3):
            resp['f'] = 1
            break
        if i==-3:
            counter2  = {'h':[0,0], 'v':[0,0], 'd':[[0, 0], [0, 0]]}
        if i==-2:
            counter3  = {'h':[0,0], 'v':[0,0], 'd':[[0, 0], [0, 0]]}
        if i==1 and (counter2['h'][0]==3 or counter2['h'][1]==3 or counter2['v'][0]==3 or counter2['v'][1]==3 or counter2['d'][0][0]==3 or counter2['d'][0][1]==3 or counter2['d'][1][0]==3 or counter2['d'][1][1]==3):
            resp['f'] = 1
            break
        if i==2 and (counter3['h'][0]==3 or counter3['h'][1]==3 or counter3['v'][0]==3 or counter3['v'][1]==3 or counter3['d'][0][0]==3 or counter3['d'][0][1]==3 or counter3['d'][1][0]==3 or counter3['d'][1][1]==3):
            resp['f'] = 1
            break
                
    return resp
        
# gera as contagens no raio 4 para cada posicao valida nas colunas
def evaluateColumns(board):
    c     = [0, 1, 2, 3, 4, 5, 6]  
    resps = [0]*7
    
    for r in range(6):
        removeList = []
        for col in c:
            if r==5 and board[r][col]==0:
                resps[col] = count(board, r, col)
                
                aux        = count(board, r-1, col)
                if aux['f'] and resps[col]['f']==0:
                    resps[col]['f'] = -1

                removeList.append(col)
            elif r+1<6:
                if board[r][col]==0 and board[r+1][col]>0:
                    resps[col] = count(board, r, col)
                    
                    if r-1>=0:
                        aux = count(board, r-1, col)
                        if aux['f'] and resps[col]['f']==0:
                            resps[col]['f'] = -1
                    
                    removeList.append(col)
        for i in removeList:
            c.remove(i)
          
    return resps
        
max_population = 15
mutation_rate = 0.09
def choice(board, gene):
    boardInfo = evaluateColumns(board)
    boardRank = [0]*7
    #print(boardInfo)
    
    maior  = -100000
    maiorI = 0
   
    for i in range(7):
        if boardInfo[i]:
            boardRank[i] = 0
            
            if boardInfo[i]['f']==1:
                return i
            elif boardInfo[i]['f']==-1:
                boardRank[i] = -5
            
            boardRank[i] += gene['b'][i] + gene['h'][0]*boardInfo[i]['h'][0] + gene['h'][1]*boardInfo[i]['h'][1] + gene['v'][0]*boardInfo[i]['v'][0] + gene['v'][1]*boardInfo[i]['v'][1] + gene['d'][0]*boardInfo[i]['d'][0] + gene['d'][1]*boardInfo[i]['d'][1]
            
            if boardRank[i]>= maior:
                maior  =  boardRank[i]
                maiorI = i
        else:
            boardRank[i] = False
    
    return maiorI
gene_fit = []

def my_agent_fit(obs, cfg):
    board = np.array(obs.board).reshape(cfg.rows, cfg.columns)

    return choice(board, gene_fit)
    #return random.choice([c for c in range(cfg.columns) if obs.board[c] == 0])
def mean_win_draw(rewards):
    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)

def fitness_fn(gene):
    global gene_fit
    
    gene_fit = gene
    fitness  = 0
    
    #fitness  += mean_win_draw(evaluate("connectx", [my_agent_fit, "random"], num_episodes=1))
    fitness  += mean_win_draw(evaluate("connectx", [my_agent_fit, "negamax"], num_episodes=8))
    fitness  += mean_win_draw(evaluate("connectx", [my_agent_fit, "rules"], num_episodes=2))
    fitness  += mean_win_draw(evaluate("connectx", [my_agent_fit, "greedy"], num_episodes=2))
    
    return fitness+1.0
def getRandGene(tam):
    return {'h':[random.uniform(0, 1), random.uniform(0, 1)], 'v':[random.uniform(0, 1), random.uniform(0, 1)], 'd':[random.uniform(0, 1), random.uniform(0, 1)], 'b':[random.uniform(0, 2) for i in range(tam)]}
def init_population(pop_number, state_length):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    state_length:  The length of each individual"""
    population = []
    
    for i in range(pop_number):
        population.append(getRandGene(state_length))

    return population
def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def select(r, population, fitnesses):
    sampler = weighted_sampler(population, fitnesses)
    
    resp = []
    for j in range(r):
        resp.append([sampler() for i in range(2)])
        
    return resp
def recombine(x, y):
    rec = {}
    
    c = random.randrange(0, 2)
    rec['h'] = x['h'][:c] + y['h'][c:]
    c = random.randrange(0, 2)
    rec['v'] = x['v'][:c] + y['v'][c:]
    c = random.randrange(0, 2)
    rec['d'] = x['d'][:c] + y['d'][c:]
    c = random.randrange(0, len(x['b']))
    rec['b'] = x['b'][:c] + y['b'][c:]
    
    return rec

def mutate(x, pmut):
    if random.uniform(0, 1) >= pmut:
        return x
    
    mut = {}
    
    c = random.randrange(0, 2)
    mut['h'] = x['h'][:c] + [random.uniform(0, 1)] + x['h'][c+1:]
    c = random.randrange(0, 2)
    mut['v'] = x['v'][:c] + [random.uniform(0, 1)] + x['v'][c+1:]
    c = random.randrange(0, 2)
    mut['d'] = x['d'][:c] + [random.uniform(0, 1)] + x['d'][c+1:]
    c = random.randrange(0, len(x['b']))
    mut['b'] = x['b'][:c] + [random.uniform(0, 2)] + x['b'][c+1:]

    return mut
ngen = 100 # maximum number of generations
# we set the threshold fitness equal to the length of the target phrase
# i.e the algorithm only terminates whne it has got all the characters correct 
# or it has completed 'ngen' number of generations
f_thres = 4

argmax = max
argmin = min

def fitness_threshold(current_best, f_thres, current_best_fit):
    if not f_thres:
        return None

    if current_best_fit >= f_thres:
        return current_best

    return None
def maiorPop(population, pupulation_fitness):
    maior  = 0
    maiorI = 0
    
    for i in range(len(population)):
        if pupulation_fitness[i]>maior:
            maior  = pupulation_fitness[i]
            maiorI = population[i]
            
    return [maiorI, maior]
    
def genetic_algorithm_stepwise(population, fitness_fn, f_thres=None, ngen=1200, pmut=0.1):
    generations_fitness = []
    
    pupulation_fitness = list(map(fitness_fn, population))

    for generation in range(ngen):
        parents = select(len(population), population, pupulation_fitness)
        
        population = [mutate(recombine(*parents[i]), pmut) for i in range(len(population))]
        #if generation:
        #    population.append(current_best)
        
        pupulation_fitness = list(map(fitness_fn, population))
        # stores the individual genome with the highest fitness in the current population
        current_best, current_best_fit = (maiorPop(population, pupulation_fitness))
        print(f'Current best: {current_best}\t\tGeneration: {str(generation)}\t\tFitness: {current_best_fit}\r', end='')
        
        # store the highet fitness per generation
        generations_fitness.append([current_best, current_best_fit])

        # compare the fitness of the current best individual to f_thres
        fittest_individual = fitness_threshold(current_best, f_thres, current_best_fit)
        
        # if fitness is greater than or equal to f_thres, we terminate the algorithm
        if fittest_individual:
            return fittest_individual, generation, generations_fitness
    return max(population, key=fitness_fn) , generation, generations_fitness
population = init_population(max_population, 7)
population[1] = {'h': [0.53972686363693, 0.044165050450831966], 'v': [0.022635003014073507, 0.6866010352817071], 'd': [0.8316090672263157, 0.6377433301389589], 'b': [1.588468329380151, 1.1538044614490195, 1.5816126357054339, 1.8251086839635926, 1.8906146697636743, 1.8043295226300544, 1.1433389106260636]}
population[0] = {'h': [0.53972686363693, 0.044165050450831966], 'v': [0.9714269240576067, 0.6866010352817071], 'd': [0.20304354584430118, 0.6377433301389589], 'b': [1.4254033462937865, 0.17929438278369947, 0.27923382858569923, 1.5705425623569804, 0.4874105225386902, 0.6238529825105752, 1.1433389106260636]}
population[2] = {'h': [0.53972686363693, 0.044165050450831966], 'v': [0.4193448263528389, 0.6866010352817071], 'd': [0.8316090672263157, 0.6377433301389589], 'b': [1.4254033462937865, 0.17929438278369947, 1.9435107126735125, 0.7395205142956387, 0.7993955389619241, 1.8043295226300544, 1.1433389106260636]}
population[3] = {'h': [0.53972686363693, 0.5542832684941604], 'v': [0.42225840018110383, 0.6406121279048417], 'd': [0.8316090672263157, 0.48628389153218454], 'b': [1.4254033462937865, 0.17929438278369947, 1.9435107126735125, 0.7395205142956387, 0.7993955389619241, 1.5044389610339026, 0.71724616758627]}
population[4] = {'h': [0.5490818885071335, 0.22364778280533126], 'v': [0.33090961553317977, 0.9515863098798619], 'd': [0.8611671606371075, 0.8580217796710116], 'b': [1.4254033462937865, 0.17929438278369947, 1.9435107126735125, 0.7395205142956387, 1.795808232174158, 1.8043295226300544, 1.1433389106260636]}
population[5] = {'h': [0.5490818885071335, 0.22364778280533126], 'v': [0.33090961553317977, 0.6866010352817071], 'd': [0.21098891177192614, 0.8580217796710116], 'b': [1.4254033462937865, 0.8845103143438735, 1.9435107126735125, 0.7395205142956387, 0.7993955389619241, 1.8043295226300544, 1.1433389106260636]}

if fit:
    solution, generations, generations_fitness = genetic_algorithm_stepwise(population, fitness_fn, f_thres, ngen, mutation_rate)
else:
    solution      = {'h': [0.53972686363693, 0.03111765260792465], 'v': [0.8142049683200521, 0.3077397943579334], 'd': [0.43633156432421916, 0.023082489442523335], 'b': [1.4254033462937865, 0.17929438278369947, 1.5816126357054339, 1.8251086839635926, 1.8906146697636743, 0.18700820417313624, 0.12891933314301962]}
def my_agent(obs, cfg):
    board = np.array(obs.board).reshape(cfg.rows, cfg.columns)

    return choice(board, solution)
env.reset()
env.run([my_agent, "negamax"]) #Agente definido em my_agent versus angente randômico.
env.render(mode="ipython", width=500, height=450)
# k is my agent
# 6 is the most right
# 1 is my agent
env.play([my_agent, None], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
def mean_win_draw(rewards):
    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)

# Run multiple episodes to estimate its performance.
vs_random = mean_win_draw(evaluate("connectx", [my_agent, "random"], num_episodes=10))
print("My Agent vs Random Agent:", vs_random)

vs_negamax = mean_win_draw(evaluate("connectx", [my_agent, "negamax"], num_episodes=10))
print("My Agent vs Negamax Agent:", vs_negamax)

vs_rules = mean_win_draw(evaluate("connectx", [my_agent, "rules"], num_episodes=10))
print("My Agent vs Rule Agent:", vs_rules)

vs_greedy = mean_win_draw(evaluate("connectx", [my_agent, "greedy"], num_episodes=10))
print("My Agent vs Greedy Agent:", vs_greedy)
print(evaluate("connectx", [my_agent, "negamax"], num_episodes=10))
import csv

seu_nome = "GUSTAVO_VILAR"

rows = [['Id', 'Predicted'],['random',vs_random],[ 'negamax', vs_negamax],[ 'rules', vs_rules],[ 'greedy', vs_greedy]]
f = open(seu_nome+'-ConnectX.csv', 'w')
with f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)

write_agent_to_file(my_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.
import sys
out = sys.stdout
submission = utils.read_file("/kaggle/working/submission.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")