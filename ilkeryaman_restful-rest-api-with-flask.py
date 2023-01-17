from flask import Flask, request, jsonify, abort, make_response
from http import HTTPStatus

app = Flask(__name__)
employees = [
    {"id": 1, "name": "Ilker Yaman"},
    {"id": 2, "name": "Serdar Burak Guneri"}
]
def get_employee(employee_id):
    employee = [employee for employee in employees if employee['id'] == employee_id]
    if len(employee) == 0:
        abort(HTTPStatus.NOT_FOUND.value)
    if request.method == "GET":
        return jsonify({'employee': employee[0]})
    elif request.method == "PUT":
        data = request.get_json()
        if data and data["name"]:
            employee[0]["name"] = data["name"]
        else:
            abort(HTTPStatus.BAD_REQUEST.value)
    else:
        abort(HTTPStatus.METHOD_NOT_ALLOWED.value)
@app.errorhandler(HTTPStatus.NOT_FOUND.value)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), HTTPStatus.NOT_FOUND.value)
if __name__ == '__main__':
    app.run(debug=True)