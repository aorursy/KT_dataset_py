#Precisamos fazer o download do ambiente de Connect X:

# Vamos precisar de um kaggle-enviroments customizado para a avaliação.

!pip install git+https://github.com/matheusgmaia/kaggle-environments

    
#Criar ambiente

from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
# -*- coding: utf-8 -*-

import random

import numpy as np

AI_MOVE = 1

HUMAN_MOVE = 2









def terminal_node(board):

    """

    Checa se o estado atual do tabuleiro é um estado terminal - 4 peças foram conectadas

    """

    score = eval_board(board)

    if score > 100000:

        return True

    elif score < -100000:

        return True

    

    return False

def eval_board(board):

    """

    Avalia o estado de um tabuleiro e retorna o seu score

    """

    def __eval_window(agent_tiles, human_tiles, score=0):

        """

        Função Heurística. Retorna um valor para cada situação de janela com 4 peças.

        """

        # Caso terminal

        if agent_tiles == 4:

            score += 1000000 # Um valor absurdamente alto define um estado terminal

        # human_tiles = 0

        elif agent_tiles == 3 and human_tiles == 0:

            score += 50

        elif agent_tiles == 2 and human_tiles == 0:

            score += 15

        elif agent_tiles == 1 and human_tiles == 0:

            score += 1



        # human_tiles = 1

        elif agent_tiles == 3 and human_tiles == 1:

            score += 8

        elif agent_tiles == 2 and human_tiles == 1:

            score += 5

        elif agent_tiles == 1 and human_tiles == 1:

            score += 0

        elif agent_tiles == 0 and human_tiles == 1:

            score -= 1



        # human_tiles = 2



        elif agent_tiles == 2 and human_tiles == 2:

            score += 0

        elif agent_tiles == 1 and human_tiles == 2:

            score -= 5

        elif agent_tiles == 0 and human_tiles == 2:

            score -= 15



        # human_tiles = 3



        elif agent_tiles == 1 and human_tiles == 3:

            score -= 8

        elif agent_tiles == 0 and human_tiles == 3:

            score -= 50



        # human_tiles = 4

        elif human_tiles == 4:

            score -= 1000000 # Um valor absurdamente alto define um estado terminal



        return score



    def __eval_hor(board, score=0):

        """

        Avalia as pontuações do tabuleiro horizontalmente

        """

        # Pegando a linha

        cols = board.shape[1]

        rows = board.shape[0]

        # Percorre todas as linhas e avalia janelas de 4 peças

        for j in range(rows):

            for i in range(cols - 3):

                window = board[j][i:i+4]

                agent_tiles = np.count_nonzero(window == AI_MOVE)

                human_tiles = np.count_nonzero(window == HUMAN_MOVE)

                score = __eval_window(agent_tiles, human_tiles, score)



        return score





    def __eval_ver(board, score=0):

        """

        Avalia as pontuações do tabuleiro verticalmente

        """

        # Pegando a linha

        cols = board.shape[1]

        rows = board.shape[0]

        # Percorre todas as colunas e avalia janelas de 4 peças

        for j in range(cols):

            for i in range(rows - 3):

                window = board[i:i+4, j]

                agent_tiles = np.count_nonzero(window == AI_MOVE)

                human_tiles = np.count_nonzero(window == HUMAN_MOVE)

                score = __eval_window(agent_tiles, human_tiles, score)



        return score



    def __eval_main_diag(board, score=0):

        """

        Avalia as pontuações do tabuleiro nas diagonais principais

        """

        cols = board.shape[1]

        rows = board.shape[0]

        # Percorre todas as diagonais princiais e avalia janelas de 4 peças

        for row in range(3, rows):

            for col in range(0, cols-3):

                window = [board[row-i][col+i] for i in range(4)]

                agent_tiles = window.count(AI_MOVE)

                human_tiles = window.count(HUMAN_MOVE)

                score = __eval_window(agent_tiles, human_tiles, score)



        return score



    def __eval_sec_diag(board, score=0):

        """

        Avalia as pontuações do tabuleiro nas diagonais secundárias

        """

        cols = board.shape[1]

        #rows = board.shape[0]

        # Percorre todas as diagonais secundárias e avalia janelas de 4 peças

        for row in range(0, 3):

            for col in range(0, cols-3):

                window = [board[row+i][col+i] for i in range(4)]

                agent_tiles = window.count(AI_MOVE)

                human_tiles = window.count(HUMAN_MOVE)

                score = __eval_window(agent_tiles, human_tiles, score)



        return score



    # FEATURES DA FUNÇÃO DE AVALIAÇÃO:

    # JANELAS HORIZONTAIS, VERTICAIS E DIAGONAIS

    # CADA FEATURE RECEBE PESO 1

    # EVAL(S) = W1*f1 + ... + Wn*fn

    score = 0

    score = __eval_hor(board, score) + __eval_ver(board, score) + __eval_main_diag(board, score) + __eval_sec_diag(board, score)

    return score



def get_board(obs, cfg):

    """

    Retorna uma matriz que representa o tabuleiro

    """

    cols = cfg.columns

    rows = cfg.rows

    shape = (rows, cols)



    board = np.array(obs.board).reshape(shape)



    return board



def gen_board(board, move, MOVE_TYPE):

    """

    Gera um tabuleiro futuro, levando em consideração a próxima jogada.

    Recebe como entrada o estado atual do tabuleiro.



    Retorna o estado do tabuleiro após a jogada.

    """





    # Checa qual é a linha certa para inserção da jogada

    new_board = board.copy()



    line = new_board.shape[0] - 1

    while new_board[line][move] != 0:

        line = line - 1



    new_board[line][move] = MOVE_TYPE



    return new_board



def get_valid_moves(board):

    """

    Retorna as jogadas válidas de acordo com o estado atual do tabuleiro

    """

    valid_moves = [col for col in range(len(board[0])) if board[0][col] == 0]

    return valid_moves





def minimax(board, depth, maximizer):

    """

    Retorna a melhor jogada, de acordo com o algoritmo minimax

    """

    if depth == 0 or terminal_node(board):

        if not get_valid_moves(board): # Sem mais jogadas válidas = empate

            return None, 0

        return None, eval_board(board) # Chegamos em uma posição vencedora. Nenhuma jogada deve ser retornada

    elif maximizer == True: # Jogada da AI

        value = -100000000

        # Expansão de todas as jogadas

        valid_moves = get_valid_moves(board)

        best_move = valid_moves[0]

        for move in valid_moves:

            resulting_board = gen_board(board, move,AI_MOVE) # Computa o tabuleiro para a próxima posição, dado a jogada atual

            new_value = minimax(resulting_board, depth - 1, False)[1] # Pega o valor do minimax

            if new_value >= value:

                value = new_value

                best_move = move

        return best_move, value

    

    else: # Jogada do humano

        value =  100000000

        valid_moves = get_valid_moves(board)

        best_move = valid_moves[0]

        for move in valid_moves:

            resulting_board = gen_board(board,move, HUMAN_MOVE) # Computa o tabuleiro para a próxima posição, dado a jogada atual

            new_value = minimax(resulting_board, depth - 1, True)[1]   # Pega o valor do minimax

            if new_value <= value:

                value = new_value

                best_move = move

                

        return best_move, value

    





import time

def my_agent(obs, cfg):  # recebe o estado atual do jogo e a configuração do jogo, retorna uma jogada

    # Gerando o tabuleiro inicial



    board = get_board(obs, cfg)

    minimax_start_time = time.time()

    best_move, _ = minimax(board, 3, True)

    minimax_end_time = time.time() - minimax_start_time

    

    

    

    return best_move
env.reset()

env.run([my_agent, "rules"]) #Agente definido em my_agent versus angente randômico.

env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.

trainer = env.train([None, "rules"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("Ação do seu agente: Coluna", my_action+1)

    observation, reward, done, info = trainer.step(my_action)

    env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
# "None" represents which agent you'll manually play as (first or second player).

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
import csv



seu_nome = "Andre Igor"



rows = [['Id', 'Predicted'],['random',vs_random],[ 'negamax', vs_negamax],[ 'rules', vs_rules],[ 'greedy', vs_greedy]]

f = open(seu_nome+'-ConnectX.csv', 'w')

with f:

    writer = csv.writer(f)

    for row in rows:

        writer.writerow(row)