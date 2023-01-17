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


with open('anne_data.csv', 'w') as f:
    f.write("%s\n" % "pos_01,pos_02,pos_03,pos_04,pos_05,pos_06,pos_07,pos_08,pos_09,pos_10,pos_11,pos_12,pos_13,pos_14,pos_15,pos_16,pos_17,pos_18,pos_19,pos_20,pos_21,pos_22,pos_23,pos_24,pos_25,pos_26,pos_27,pos_28,pos_29,pos_30,pos_31,pos_32,pos_33,pos_34,pos_35,pos_36,pos_37,pos_38,pos_39,pos_40,pos_41,pos_42,coluna")
    for i in range(10000):
        trainer = env.train([None, "random"])

        observation = trainer.reset()
        while not env.done:
            my_action = my_agent(observation, env.configuration)
            print("Ação do seu agente: Coluna", my_action+1)
            print(observation.board)
            a = observation.board.copy()
            a.append(my_action+1)
            #print(observation.board)
            
            data = str(a)[1:-1]
            
            #data =print(', '.join(observation.board))
            f.write("%s\n" % data) # observation.board)
            observation, reward, done, info = trainer.step(my_action)
            env.render(mode="ipython", width=100, height=90, header=False, controls=False)
            #env.render()
        env.render()


# "None" represents which agent you'll manually play as (first or second player).
env.play([None, my_agent], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
from sklearn import tree
from sklearn.metrics import confusion_matrix
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score # to score our model

X_full = pd.read_csv('../input/annedata/anne_data.csv') 
X = X_full.drop(columns=['coluna'])

y = X_full['coluna']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
preds = clf.predict(X_valid)
accuracy_score(y_valid, preds)

confusion_matrix(y_valid, preds)
#Sua vez
def my_agent(obs, cfg):
    y_pred = clf.predict([obs.board])
    return y_pred.tolist()[0]


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

seu_nome = "ANNE_MOREIRA"

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