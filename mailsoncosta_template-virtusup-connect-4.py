#Precisamos fazer o download do ambiente de Connect X:
# Vamos precisar de um kaggle-enviroments customizado para a avaliação.
!pip install git+https://github.com/matheusgmaia/kaggle-environments
    
#Criar ambiente
from kaggle_environments import evaluate, make, utils
import numpy as np

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
def check_winner_piece(board, piece):
    for row in range(board.shape[0]):
        for column in range(board.shape[1]):
            if (column + 3) < board.shape[1] and board[row,column] != 0:
                if board[row,column] == board[row,column + 1] and \
                   board[row,column + 1] == board[row,column + 2] and \
                   board[row,column + 2] == board[row,column + 3]:
                    return board[row,column] == piece

            if (row + 3) < board.shape[0] and board[row,column] != 0:
                if board[row,column] == board[row + 1,column] and \
                   board[row + 1,column] == board[row + 2,column] and \
                   board[row + 2,column] == board[row + 3,column]:
                    return board[row,column] == piece

            if (column + 3) < board.shape[1] and (row + 3) < board.shape[0] and board[row,column] != 0:
                if board[row,column] == board[row + 1,column + 1] and \
                   board[row + 1,column + 1] == board[row + 2,column + 2] and \
                   board[row + 2,column + 2] == board[row + 3,column + 3]:
                    return board[row,column] == piece

            if (column + 3) < board.shape[1] and (row + 3) < board.shape[0] and board[row + 3,column] != 0:
                if board[row + 3,column] == board[row + 2,column + 1] and \
                   board[row + 2,column + 1] == board[row + 1,column + 2] and \
                   board[row + 1,column + 2] == board[row,column + 3]:
                    return board[row + 3,column] == piece
    return False

def drop_piece(board, column, piece):  
    temp_board = board.copy()
    if temp_board[0,column] == 0:
        best_mov1e = 0
        for y in range(temp_board.shape[0]):
            if temp_board[y,column] == 0:
                best_move = y
        temp_board[best_move, column] = piece
        
        return temp_board

def calculate_pontuation(piece, my_piece, good_score, bad_score):
    return good_score if piece == my_piece else bad_score
    
def greedy_score(board, column, my_piece):
    score = 0
    good_score_2x = 61
    good_score_1x = 29
    bad_score_2x = 33
    bad_score_1x = 17
    for row in range(board.shape[0]):
        if (column + 3) < board.shape[1] and board[row,column] != 0:
            if board[row,column] == board[row,column + 1] and \
               board[row,column + 1] == board[row,column + 2]:
                score += calculate_pontuation(board[row,column + 2], my_piece, good_score_2x, bad_score_2x)
            elif board[row,column] == board[row,column + 1]:
                score += calculate_pontuation(board[row,column], my_piece, good_score_1x, bad_score_1x)

        if (row + 3) < board.shape[0] and board[row,column] != 0:
            if board[row,column] == board[row + 1,column] and \
               board[row + 1,column] == board[row + 2,column]:
                score += calculate_pontuation(board[row + 2,column], my_piece, good_score_2x, bad_score_2x)
            elif board[row,column] == board[row + 1,column]:
                score += calculate_pontuation(board[row,column], my_piece, good_score_1x, bad_score_1x)

        if (column + 3) < board.shape[1] and (row + 3) < board.shape[0] and board[row,column] != 0:
            if board[row,column] == board[row + 1,column + 1] and \
               board[row + 1,column + 1] == board[row + 2,column + 2]:
                score += calculate_pontuation(board[row + 2,column + 2], my_piece, good_score_2x, bad_score_2x)
            elif board[row,column] == board[row + 1,column + 1]:
                score += calculate_pontuation(board[row,column], my_piece, good_score_1x, bad_score_1x)

        if (column + 3) < board.shape[1] and (row + 3) < board.shape[0] and board[row + 3,column] != 0:
            if board[row + 3,column] == board[row + 2,column + 1] and \
               board[row + 2,column + 1] == board[row + 1,column + 2]:
                score += calculate_pontuation(board[row + 1,column + 2], my_piece, good_score_2x, bad_score_2x)
            elif board[row + 3,column] == board[row + 2,column + 1]:
                score += calculate_pontuation(board[row,column], my_piece, good_score_1x, bad_score_1x)

    return score

def drop_on_column(current_board, original_board):
    current_board = current_board.flatten()
    original_board = original_board.flatten()

    for index in range(len(current_board)):
        if current_board[index] != original_board[index]:
            return index//7

def pick_best_move(board, valid_locations, my_piece, opp_piece):
    best_score = -10000
    best_locations = [valid_locations[0]]
    save_opp_best = []
    save_my_best = []
    for location in valid_locations:
        my_win_board = drop_piece(board, location, my_piece)
        opp_win_board = drop_piece(board, location, opp_piece)
        if check_winner_piece(my_win_board, my_piece):
            boards = [my_win_board]
            save_my_best.append(location)

        elif check_winner_piece(opp_win_board, opp_piece):
            boards = [opp_win_board]
            save_opp_best.append(location)

        else:
            # calcula melhor escolha - guloso
            current_score = greedy_score(board, location, my_piece)

            if current_score > best_score:
                best_score = current_score
                best_locations = [location]
    
            elif current_score == best_score:
                best_locations.append(location)

    if len(save_my_best) != 0:
        return random.choice(save_my_best)
    elif len(save_opp_best) != 0:
        return random.choice(save_opp_best)

    return random.choice(best_locations)

def connectFour_agent(obs, cfg):
    board = np.reshape(obs["board"],(6,7))
    #print("board: ", board)
    valid_locations = [column for column in range(cfg.columns) if obs.board[column] == 0] # todo valor igual a zero
    my_piece = obs["mark"]
    opp_piece = 2
    if my_piece == 2:
        opp_piece = 1

    return pick_best_move(board, valid_locations, my_piece, opp_piece) 



# "None" represents which agent you'll manually play as (first or second player).
env.play([None, connectFour_agent], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
env.reset()
# env.run([connectFour_agent, "random"]) #Agente definido em connectFour_agent versus angente randômico.
# env.run([connectFour_agent, connectFour_agent]) #Agente definido em connectFour_agent versus connectFour_agent.
# env.run([connectFour_agent, "negamax"]) #Agente definido em connectFour_agent versus angente negamax.
env.run([connectFour_agent, "rules"]) #Agente definido em connectFour_agent versus angente rules.
# env.run([connectFour_agent, "greedy"]) #Agente definido em connectFour_agent versus angente greedy.
env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()

while not env.done:
    my_action = connectFour_agent(observation, env.configuration)
    print("Ação do seu agente: Coluna", my_action+1)
    observation, reward, done, info = trainer.step(my_action)
    env.render(mode="ipython", width=100, height=90, header=False, controls=False)
env.render()
def mean_win_draw(rewards):
    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)

# Run multiple episodes to estimate its performance.
vs_random = mean_win_draw(evaluate("connectx", [connectFour_agent, "random"], num_episodes=10))
print("My Agent vs Random Agent:", vs_random)

vs_negamax = mean_win_draw(evaluate("connectx", [connectFour_agent, "negamax"], num_episodes=10))
print("My Agent vs Negamax Agent:", vs_negamax)

vs_rules = mean_win_draw(evaluate("connectx", [connectFour_agent, "rules"], num_episodes=10))
print("My Agent vs Rule Agent:", vs_rules)

vs_greedy = mean_win_draw(evaluate("connectx", [connectFour_agent, "greedy"], num_episodes=10))
print("My Agent vs Greedy Agent:", vs_greedy)
import csv

seu_nome = "Mailson Nascimento Costa"

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