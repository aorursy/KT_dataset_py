# Necessary imports

from random import randrange

from multiprocessing import Process, Queue

from kaggle_environments import make, evaluate
board_size = 11

agent_count = 4

nb_processes = 3





def play_game(game_id, queue):

    environment = make("halite", configuration={"size": board_size, 'randomSeed': randrange((1 << 32) - 1)}, debug=True)

    environment.reset(agent_count)

    game = environment.run(["random", "random", "random", "random"])

    print("Finished game", game_id)

    queue.put(environment)





queue = Queue()

processes = []



for p in range(1, nb_processes + 1):

    process = Process(target=play_game, args=(p, queue))

    processes.append(process)

    process.start()



for _ in range(nb_processes):

    queue.get().render(mode="ipython", width=500, height=400)