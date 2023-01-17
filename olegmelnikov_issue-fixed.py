import kagglegym

env = kagglegym.make()

o = env.reset()

done = False

n_max_steps=5



while not done and n_max_steps > 0:

    print(type(o.features).__name__, type(o.target).__name__, type(o.train).__name__)

    o, reward, done, info = env.step(o.target)

    n_max_steps -= 1