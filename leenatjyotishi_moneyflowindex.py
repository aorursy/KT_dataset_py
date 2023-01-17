import pandas as pd
def moneyFlowIndex( filename, n ):
    dt=pd.read_csv("../input/moneyflowindexleena/sample.csv") 
    # View dataframe
   
    Open = dt["Open"]
    High = dt["High"]
    Low = dt["Low"]
    Close = dt["Close"]
    Volume = dt["Volume"]

    # Create dataframe
    df = pd.DataFrame(dt) 
    # Calculate TypicalPrice
    TypicalPrice = (High + Low +Close)/3
    df['Typical Price'] = TypicalPrice 
    df['Positive Money Flow']=0
    df['Negative Money Flow']=0

    MoneyFlow=TypicalPrice * Volume

    # Calculate Positive/Negative Money Flow
    df.loc[df['Typical Price'] > df['Typical Price'].shift(1), 'Positive Money Flow'] = MoneyFlow 
    df.loc[df['Typical Price'] < df['Typical Price'].shift(1), 'Negative Money Flow'] = MoneyFlow

    # Calculate Positive/Negative Money Flow sum
    df['Positive Money Flow Sum'] = df['Positive Money Flow'].rolling(n).sum().astype('float64').round(2) 
    df['Negative Money Flow Sum'] = df['Negative Money Flow'].rolling(n).sum().astype('float64').round(2) 

    # Calculate Money Flow Index
    MoneyRatio =df['Positive Money Flow Sum']/df['Negative Money Flow Sum'] 
    MoneyFlowIndex=((MoneyRatio) / (1+MoneyRatio)) *100
    df['Money Flow Index'] = MoneyFlowIndex 
    df.to_csv('money flow index' + str(n) +'.csv')
    return;

moneyFlowIndex('sample.csv',16)