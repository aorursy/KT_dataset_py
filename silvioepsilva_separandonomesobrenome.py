import pandas as pd
df = pd.DataFrame({'Nome': ['Dayana Guedes', 'Camila Fernandes', 'Felipe Luiz Guimarães', 'Dayana Lima', 'Jander Guimarães', 'Paulo de Souza', 
                            'Suzilaine de Oliveira Guimarães', 'Marcos Paulo Filho', 'Oswaldo Neto', 'Gabriela Martins']
                  })
df
df['Primeiro Nome'] = df['Nome'].str.split(n=1).str[0]
df['Sobrenome'] = df['Nome'].str.split(n=1).str[1]
df