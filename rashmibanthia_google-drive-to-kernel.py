#Keep Internet on



#https://drive.google.com/open?id=1FAkQtAZRoD3kyMNwD9LQTMsDtAFe3CY4  

#This is the id -> 1FAkQtAZRoD3kyMNwD9LQTMsDtAFe3CY4 Replace it twice in the following cell

#crawl-300d-2M.pkl   
!wget wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FAkQtAZRoD3kyMNwD9LQTMsDtAFe3CY4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FAkQtAZRoD3kyMNwD9LQTMsDtAFe3CY4" -O crawl-300d-2M.pkl  && rm -rf /tmp/cookies.txt
!ls -l