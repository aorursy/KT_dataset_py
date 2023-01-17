import pandas as pd
df = pd.DataFrame({'Nome': ['Lucas', 'Pedro', 'Maria', 'Paulo', 'Eliana'],
                  'Salário': [3200.45, 'R$ 1550.00', 4200.80, 'R$1700.00', 4500.00]
                  })
df
df.info()
df['Salário'] = df['Salário'].replace('[R$,]', '', regex=True).astype('float')
df.info()
df