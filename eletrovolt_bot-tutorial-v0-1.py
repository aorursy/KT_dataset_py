# Configura o ambiente disponibilizado pelo Kaggle do Halite IV
from kaggle_environments import evaluate, make
# Declara a variável que armazena informações sobre o ambiente do jogo
env = make("halite", configuration={ "episodeSteps": 400 }, debug=True)

# Imprime as configurações do ambiente
print (env.configuration)
%%writefile submission.py

from kaggle_environments.envs.halite.helpers import *
from random import choice

# Função principal do agente: recebe observação e configuração e retorna uma ação.
def agent(obs, config):
    # Constrói um objeto Board a partir da observação e configuração.
    board = Board(obs, config)
    # Atribui à variável "me" o objeto do jogador da vez (no caso, o próprio agente)
    me = board.current_player
    
    # Itera por todos os navios disponíveis para o agente
    for ship in me.ships:
        # Configura a próxima ação do navio para ser uma escolha aleatória entre norte, sul, leste, oeste ou não fazer nada.
        ship.next_action = choice([ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None])
    
    # Itera por todos os estaleiros disponíveis para o agente
    for shipyard in me.shipyards:
        # Define que a próxima ação deve ser não fazer nada.
        shipyard.next_action = None
    
    # Retorna todas as ações definidas para cada um dos navios e estaleiros.
    return me.next_actions
# Utiliza do ambiente do jogo para rodar o bot contra outros 3 agentes aleatórios
env.run(["/kaggle/working/submission.py", "random","random","random"])
# Renderiza uma caixa de visualização com o jogo logo abaixo
env.render(mode="ipython", width=800, height=600)

# É possível ver as ações do bot no jogo abaixo (bot amarelo)