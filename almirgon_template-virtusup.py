#Precisamos fazer o download do ambiente de Connect X:
# Vamos precisar de um kaggle-enviroments customizado para a avaliação.
!pip install git+https://github.com/matheusgmaia/kaggle-environments
    
import inspect
import os
import numpy as np
import scipy
from random import choice

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
env.specification
def meu_agente(observation, configuration):
    linhas = configuration.rows #Número de linhas no tabuleiro
    colunas = configuration.columns #Número de colunas no tabuleiro
    inarow = configuration.inarow #Número de damas necessárias para ganhar
    colunas_por_prioridade = [3, 4, 2, 5, 1, 0, 6] #Lista de colunas prioritarias, da mais importante para a menos
    
    "Função responsável por retornar o valor do primeiro elemento da lista de colunas_por_prioridade"
    def seleciona_primeiro(observation):
        for i in colunas_por_prioridade:
            if observation.board[i] == 0:
                return i
    
    """"Função responsavel por observar o tabuleiro de 4 formas diferentes de acordo com 
    as linhas, colunas, e valor para o ganho(inarow) passadas como parametros afim de encontrar o próximo passo
    vencedor apra um jogador específico e por fim retorna uma 
    coluna se uma etapa vencedora for possivel, se não retorna um inteiro (-99)"""
    def passo_vencedor(observation, jogador, linhas, colunas, inarow):
       
        #Checa a possibilidade horizontal
        for i in reversed(range(linhas)):
            for j in reversed(range(colunas - inarow)):
                if observation.board[j + colunas * i] == jogador \
                        and observation.board[j + colunas * i + 1] == jogador \
                        and observation.board[j + colunas * i + 2] == jogador:
                    # Teste se estiver nas posições (4, 11, 18, 25, 32, 39)
                    if j != (colunas - inarow + 1) + colunas * i and observation.board[j + colunas * i + 3] == 0:
                        return j + 3
                    # Teste se estiver nas posições (0, 7, 14, 21, 28, 35)
                    elif j != colunas * i and observation.board[j + colunas * i - 1] == 0:
                        return j - 1

        #Checa a possibilidade vertical
        for i in range(colunas):
            for j in reversed(range(linhas - inarow)):
                if observation.board[i + colunas * (j + 1)] == jogador \
                        and observation.board[i + colunas * (j + 2)] == jogador \
                        and observation.board[i + colunas * (j + 3)] == jogador \
                        and observation.board[i + colunas * (j)] == 0:
                    return i

        #Checa a possibilidade diagonal
        for i in range(linhas - inarow - 4):
            for j in range(colunas - inarow - 1):
                if observation.board[j + colunas * i] == jogador \
                        and observation.board[j + 1 + colunas * (i + 1)] == jogador \
                        and observation.board[j + 2 + colunas * (i + 2)] == jogador:
                    # Teste se estiver nas posições (4, 11, 18, 25, 32, 39)
                    # e (35, 36, 37, 38, 39, 40, 41)
                    if j != (colunas - inarow + 1) + colunas * i \
                            and j not in range(colunas * (linhas - 1), colunas * linhas) \
                            and observation.board[j + 3 + colunas * (i + 3)] == 0:
                        return j + 3
                    # Teste se estive nas posições (position 0, 7, 14, 21, 28, 35) e (0, 1, 2, 3, 4, 5, 6)
                    elif j != colunas * i and j not in range(colunas) and observation.board[
                        j - 1 + colunas * (i - 1)] == 0:
                        return j - 1

        #Checa a possibilidade diagonal invertida
        for i in range(linhas - inarow - 1):
            for j in range(inarow - 1, colunas):
                if observation.board[j + colunas * i] == jogador \
                        and observation.board[j + 1 + colunas * (i + 1)] == jogador \
                        and observation.board[j + 2 + colunas * (i + 2)] == jogador:
                    # Teste se estive nas posições(4, 11, 18, 25, 32, 39)
                    if j != (colunas - inarow + 1) + colunas * i \
                            and observation.board[j + 3 + colunas * (i + 3)] == 0:
                        return j + 3
                    # Teste se estive nas posições (0, 7, 14, 21, 28, 35) e (0, 1, 2, 3, 4, 5, 6)
                    elif j != colunas * i and j not in range(colunas) and observation.board[
                        j - 1 + colunas * (i - 1)] == 0:
                        return j - 1
        else:
            return -99
    
    "Função responsável por retornar uma tupla com um número e a referencia para o jogador iniciante"
    def pega_numero(observation):
        contador1 = 0
        contador2 = 0
        for i in observation.board:
            if i == 1:
                contador1 += 1
            elif i == 2:
                contador2 += 1
        if contador1 > contador2:
            jogador_iniciante = 1
        elif contador1 < contador2:
            jogador_iniciante = 2
        else:
            jogador_iniciante = 0
        return contador1 + contador2, jogador_iniciante  

    p_num, jogador_iniciante = pega_numero(observation=observation)

    if p_num < 5:
        return seleciona_primeiro(observation=observation)
    else:
        # Verifica a possibilidade de ganhar
        vitoria_propria = passo_vencedor(observation=observation, jogador=1, linhas=linhas, colunas=colunas,
                               inarow=inarow)
        if vitoria_propria >= 0:  
            return vitoria_propria

        # Verifica a possibilidade do jogador oposto ganhar
        vitoria_oposta = passo_vencedor(observation=observation, jogador=2, linhas=linhas, colunas=colunas,
                                    inarow=inarow)
        if vitoria_oposta >= 0:
            return vitoria_oposta
        else:
            return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = meu_agente(observation, env.configuration)
    print("Ação do seu agente: Coluna", my_action)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
def mean_win_draw(rewards):
    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)

# Run multiple episodes to estimate its performance.
vs_random = mean_win_draw(evaluate("connectx", [meu_agente, "random"], num_episodes=10))
print("My Agent vs Random Agent:", vs_random)

vs_negamax = mean_win_draw(evaluate("connectx", [meu_agente, "negamax"], num_episodes=10))
print("My Agent vs Negamax Agent:", vs_negamax)

vs_rules = mean_win_draw(evaluate("connectx", [meu_agente, "rules"], num_episodes=10))
print("My Agent vs Rule Agent:", vs_rules)

vs_greedy = mean_win_draw(evaluate("connectx", [meu_agente, "greedy"], num_episodes=10))
print("My Agent vs Greedy Agent:", vs_greedy)
import csv

seu_nome = "Almir"

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