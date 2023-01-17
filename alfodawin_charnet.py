!rm -rf CharNet
!git clone https://github.com/ClashLuke/CharNet
!python -m pip install -q --upgrade pydot graphviz ipython ipykernel tensorflow-addons
!cp CharNet/tinyshakespeare.txt .
!rm -rf mlp_weights
!mkdir mlp_weights
from CharNet import CharNet
network = CharNet({'batch_size':           256,
                   'inputs':               64,
                   'embedding':            True,
                   'neurons_per_layer':    8,
                   'layer_count':          8,
                   'dropout':              0.5,
                   'generated_characters': 1024
                  })
network.train('tinyshakespeare.txt', workers=2, epochs=32)