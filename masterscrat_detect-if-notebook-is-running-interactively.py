def is_interactive():

   return 'runtime' in get_ipython().config.IPKernelApp.connection_file



print('Interactive?', is_interactive())
import os

def is_interactive():

   return 'SHLVL' not in os.environ



print('Interactive?', is_interactive())