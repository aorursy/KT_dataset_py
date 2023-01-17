from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hey <b>there</b>!<br>{ 6 * 7}"

app.run()

