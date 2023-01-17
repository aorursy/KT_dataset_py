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
    print(observation)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
# "None" represents which agent you'll manually play as (first or second player).
env.play([None, my_agent], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
!pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
# Definindo as variáveis do sistema e as funções de pertinência

# Perigo do adversário fechar uma sequencia
danger = np.arange(0, 4, 1)

# Change do agente fechar uma sequência
winnable = np.arange(0, 4, 1)

danger_vl = fuzz.trimf(danger, [0, 0, 1])
danger_lo = fuzz.trimf(danger, [0, 1, 2])
danger_me = fuzz.trimf(danger, [1, 2, 3])
danger_hi = fuzz.trimf(danger, [2, 3, 3])

win_vl = fuzz.trimf(winnable, [0, 0, 1])
win_lo = fuzz.trimf(winnable, [0, 1, 2])
win_me = fuzz.trimf(winnable, [1, 2, 3])
win_hi = fuzz.trimf(winnable, [2, 3, 3])

# Função de pertinência de saida
actions = np.arange(0, 3, 1)
act_lo = fuzz.trimf(actions, [0,0,1])
act_me = fuzz.trimf(actions, [0,1,2])
act_hi = fuzz.trimf(actions, [1,2,2])

danger_funcs = [danger_vl, danger_lo, danger_me, danger_hi]
win_funcs = [win_vl, win_lo, win_me, win_hi]
actions_funcs = [act_lo, act_me, act_hi]
def plot_graph(pert_func, variable, title="gráfico"):

    fig, ax0 = plt.subplots(nrows=1, figsize=(6, 4))

    colors = ['b','g','r','c','m','y','k','w']

    for i in range(len(pert_func)):
        ax0.plot(variable, pert_func[i], colors[i % len(colors)], linewidth=1.5)
        
    ax0.set_title(title)
    ax0.legend()
    
plot_graph(actions_funcs, actions, title="Gráfico da saída")
def get_matrix(obs):
    elements = obs['board']
    matrix = []
    line = []
    
    for i in range(len(elements)):
        line.append(elements[i])
    
        if (i + 1) % 7 == 0:
            matrix.append(line)
            line = []
        
    return matrix
def getValueFromPoint(matrix, point, value):
    sequencies = []
    front = getFrontSequence(matrix, point, value)
    back = getBackSequence(matrix, point, value)
    down = getDownSequence(matrix, point, value)
    dTL = getDTopLeft(matrix, point, value)
    dTR = getDTopRight(matrix, point, value)
    dDL = getDDownLeft(matrix, point, value)
    dDR = getDDownRight(matrix, point, value)
    
    sequencies.append(front)
    sequencies.append(back)
    sequencies.append(down)
    sequencies.append(dTL)
    sequencies.append(dTR)
    sequencies.append(dDL)
    sequencies.append(dDR)
    sequencies.append(front + back)
    sequencies.append(dTL + dDR)
    sequencies.append(dTR + dDL)
    return sequencies

def getFrontSequence(matrix, point, value):
    y, x = point

    init = x + 1
    end = min(3, 6 - x)
    count = 0
    for i in range(init, init + end):
        if matrix[y][i] == value:
            count += 1

        else:
            break

    return count

def getBackSequence(matrix, point, value):
    y, x = point

    init = x - 1
    end = min(-1, x - 4)
    count = 0
    for i in range(init, end, -1):
        if matrix[y][i] == value:
            count += 1

        else:
            break
    
    return count


def getDownSequence(matrix, point, value):
    y, x = point

    init = y + 1
    end = min(3, 5 - y)
    count = 0
    for i in range(init, init + end):
        if matrix[i][x] == value:
            count += 1

        else:
            break
    
    return count

def getDTopRight(matrix, point, value):
    y, x = point

    if y == 0 or x == 6: return 0

    distance = min(3, min(6 - x, y + 1))
    y = y - 1
    x = x + 1
    count =  0
    for i in range(distance):
        if matrix[y - i][x + i] == value:
            count += 1

        else:
            break
    
    return count

def getDDownRight(matrix, point, value):
    y, x = point

    if y == 5 or x == 6: return 0

    distance = min(3, min(6 - x, 5 - y))
    y = y + 1
    x = x + 1
    count =  0
    for i in range(distance):
        if matrix[y + i][x + i] == value:
            count += 1

        else:
            break
    
    return count

def getDDownLeft(matrix, point, value):
    y, x = point

    if y == 5 or x == 0: return 0

    distance = min(3, min(x, 5 - y))
    y = y + 1
    x = x - 1
    count =  0
    for i in range(distance):
        if matrix[y + i][x - i] == value:
            count += 1

        else:
            break
    
    return count

def getDTopLeft(matrix, point, value):
    y, x = point

    if y == 0 or x == 0: return 0

    distance = min(3, min(x, y + 1))
    y = y - 1
    x = x - 1
    count =  0
    for i in range(distance):
        if matrix[y - i][x - i] == value:
            count += 1

        else:
            break
    
    return count
def calculateAllPoints(value, matrix):
    values = []
    for i in range(7):
        for j in range(5, -1, -1):
            if matrix[j][i] == 0:
                if j == 5 or matrix[j + 1][i] != 0:
                    values.append([i, getValueFromPoint(matrix, (j, i), value)])
                    
    return values

def calculateMaxScore(values):
    maxV = -1
    column = -1
    for value in values:
        x = value[0]
        scores = value[1]
        bigger = max(scores)
        if bigger >= maxV:
            maxV = bigger
            column = x
            
    return maxV, column

# REGRAS NEBULOSAS

# 1. Se a Winnable for alto e vocẽ for o jogador atacante, joga ofensivamente
# 2. Se o Danger for alto/medio, joga defensivamente
# 3. Caso contrário, joga na coluna com maior score de chance

def defineAction(matrix, mark):
    
    userMark = mark
    enemyMark = 2
    
    if userMark == 2:
        enemyMark = 1

    
    userValues = calculateAllPoints(userMark, matrix)
    enemyValues = calculateAllPoints(enemyMark, matrix)
    
    win_score, attackColumn = calculateMaxScore(userValues)
    danger_score, defenseColumn = calculateMaxScore(enemyValues)
    
    danger_level_vl = fuzz.interp_membership(danger, danger_vl, danger_score)
    danger_level_lo = fuzz.interp_membership(danger, danger_lo, danger_score)
    danger_level_me = fuzz.interp_membership(danger, danger_me, danger_score)
    danger_level_hi = fuzz.interp_membership(danger, danger_hi, danger_score)

    # Checa regra 1
    active_rule_1 = fuzz.interp_membership(winnable, win_hi, win_score)
    win_play = np.fmin(active_rule_1, np.fmin(act_hi, (userMark == 1)))

    # Checa regra 2
    active_rule_2 = np.fmax(danger_level_me, danger_level_hi)
    defense_play = np.fmin(active_rule_2, act_lo)

    # Checa regra 3
    active_rule_3 = abs(1 - active_rule_2)
    normal_play = np.fmin(active_rule_3, act_me)

    aggregated = np.fmax(win_play,np.fmax(defense_play, normal_play))

    action = fuzz.defuzz(actions, aggregated, 'centroid')
    
    if action <= 0.5:
        return defenseColumn
    
    else:
        return attackColumn    
def my_agent(obs, cfg):
    matrix = get_matrix(obs)
    return defineAction(matrix, obs['mark'])
    
# env.play([None, my_agent], width=500, height=450)
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
import csv

seu_nome = "Jessé Souza Cavalcanti Neto"

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