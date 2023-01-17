%%writefile submission.py
#global params:
params={'debug':False, 
        'num_ships':50}
initd=False    

def init(observation,config):
    try:
        print('I am player id {}'.format(observation.player))
        my_id=observation.player
        myparams=config.myparams[str(my_id)]
        print('default params',params)
        params.update(myparams)
        print('got myparams',myparams)
        print('new global params',params)
    except AttributeError as ex:
        print('did not find any myparams, using defaults')

    
def agent(observation, config):
    #Just a sample agent that does nothing
    global initd
    if not initd:
        init(observation,config)
        initd=True
    return None
from kaggle_environments import make
import kaggle_environments
import json
env = make("halite", debug=True, configuration={'randomSeed': 1})
env.configuration.myparams={}
env.configuration.myparams['0']={'debug':True, 'num_ships':100}
env.run(['submission.py','random','random','random'])
x=env.render(mode='json')
res=json.loads(x)
rewards=res['rewards']
print('rewards were',rewards)

from kaggle_environments import make
import kaggle_environments
import json
env = make("halite", debug=True, configuration={'randomSeed': 1})
#env.configuration.myparams={}
#env.configuration.myparams['0']={'debug':True, 'num_ships':100}
env.run(['submission.py','random','random','random'])
x=env.render(mode='json')
res=json.loads(x)
rewards=res['rewards']
print('rewards were',rewards)