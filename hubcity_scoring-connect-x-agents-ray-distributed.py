import ray

import json

from kaggle_environments.utils import structify



def make_remote(f):

    # CHANGE THIS IF YOU ARE USING A GPU

    @ray.remote(num_gpus = 0.0)

    def fr(*args, **kargs):

        return f(*args, **kargs)

    return fr

    

def win_loss_draw(score):

    if score>0: 

        return 'win'

    if score<0: 

        return 'loss'

    return 'draw'



def score(agent, max_lines = 1000):

    #Scores a connect-x agent with the dataset

    print("scoring ",agent)

    agent = make_remote(agent)

    count = 0

    good_move_count = 0

    perfect_move_count = 0

    observation = structify({'mark': None, 'board': None})

    moves_taken = []

    filename = "/kaggle/input/1k-connect4-validation-set/refmoves1k_kaggle"

    with open(filename) as f:

        for line in f:

            count += 1

            data = json.loads(line)

            observation.board = data["board"]

            # find out how many moves are played to set the correct mark.

            ply = len([x for x in data["board"] if x>0])

            if ply&1:

                observation.mark = 2

            else:

                observation.mark = 1

            

            #call the agent

            agent_move = agent.remote(observation,env.configuration)

            moves_taken.append(agent_move)

 

            if count == max_lines:

                break

           

    moves_taken = ray.get(moves_taken)

    count = 0

    with open(filename) as f:

        for idx, line in enumerate(f):

            count += 1

            agent_move = moves_taken[idx]

            data = json.loads(line)

            moves = data["move score"]

            perfect_score = max(moves)

            perfect_moves = [ i for i in range(7) if moves[i]==perfect_score]



            if(agent_move in perfect_moves):

                perfect_move_count += 1



            if win_loss_draw(moves[agent_move]) == win_loss_draw(perfect_score):

                good_move_count += 1



            if count == max_lines:

                break



        print("perfect move percentage: ",perfect_move_count/count)

        print("good moves percentage: ",good_move_count/count)

# Score the 2 built in agents

from kaggle_environments import make





ray.init(ignore_reinit_error=True)



env = make("connectx")

score(env.agents["random"],100)  

score(env.agents["negamax"],100)



ray.shutdown()
